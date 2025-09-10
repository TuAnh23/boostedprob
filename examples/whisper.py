from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import Audio, load_dataset
import torch
import boostedprob


# Load model and processor
model_id = "openai/whisper-large-v3"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
processor = WhisperProcessor.from_pretrained(model_id)
model = WhisperForConditionalGeneration.from_pretrained(model_id).to(device)
forced_decoder_ids = processor.get_decoder_prompt_ids(
    language="english", task="translate"
)

# Load and preprocess data
ds = load_dataset("google/fleurs", "vi_vn", split="validation", trust_remote_code=True)
ds = ds.cast_column("audio", Audio(sampling_rate=16_000))
input_speech = ds[0]["audio"]
input_features = processor(input_speech["array"], sampling_rate=input_speech["sampling_rate"], return_tensors="pt").input_features.to(device)
# Add gold English translation column 
ds_en = load_dataset("google/fleurs", "en_us", split="validation", trust_remote_code=True)
trans_by_id = {x['id']:x['raw_transcription'] for x in ds_en}
trans_sorted = [trans_by_id[ds[i]['id']] for i in range(len(ds))]
ds = ds.add_column("translation", trans_sorted)

# Generate translation
with torch.no_grad():
    output = model.generate(
        input_features, 
        forced_decoder_ids=forced_decoder_ids, 
        num_beams=1, 
        do_sample=False,
        return_dict_in_generate=True, 
        output_hidden_states=True
    )
    predicted_tokens_ids = output['sequences'][:, 4:]  # Skip the first 4 tokens: 1 eos and 3 forced decoder ids

    # Get the full (log) probability distribution at each output step
    last_hidden_states = [output['decoder_hidden_states'][i][-1] for i in range(len(output['decoder_hidden_states']))]  # except for the first item, all entries have shape [batch_size,1,hidden_size]
    last_hidden_states[0] = last_hidden_states[0][:,-1:,:]  # first entry also contains the hidden states of the special tokens which we ignore, so we only get the last entry
    last_hidden_states = torch.cat(last_hidden_states, dim=1)  # [batch_size,nr_tokens,hidden_size]
    unembedding_logits = model.proj_out(last_hidden_states)  # could have gotten to this step simply with output['scores'], but the values there look errornous with many -inf values
    full_log_probs = torch.nn.functional.log_softmax(unembedding_logits, dim=-1)  # [batch_size,nr_tokens,vocab_size]

# Calculate normal prob and boostedprob
scores = torch.gather(full_log_probs, -1, predicted_tokens_ids.unsqueeze(-1)).exp()
boosted_scores = boostedprob.calculate_boostedprob(full_log_probs, predicted_tokens_ids)

# Display values
print(f"Transcription: {ds[0]['raw_transcription']}")
print(f"Gold translation: {ds[0]['translation']}")
print(f"Model translation: {processor.decode(predicted_tokens_ids[0], skip_special_tokens=True)}")
out_tokens = processor.tokenizer.convert_ids_to_tokens(predicted_tokens_ids[0])
for tok, score, boosted_score in zip(out_tokens, scores[0], boosted_scores[0]):
    print(tok,score.item(),boosted_score.item())