1. Run ```generate_tokenized_data.py``` to generate tokenized data.
    You can split the the tokenized data to ```train_tokenized.jsonl``` and ```dev_tokenized.jsonl```
2. Run ```train.sh``` to train the retriever.
3. Run ```convert_to_plm``` to convert checkpoint to huggingface model.