# BERT NER

Use google BERT to do NER, based on [REF]

# Requirements

-  `python3`
- `pip3 install -r requirements.txt`

# Run

`python run_ner.py --data_dir=data/mio --bert_model=bert-base-cased --task_name=ner --output_dir=out_base --max_seq_length=128 --do_train --num_train_epochs 5 --do_eval --warmup_proportion=0.1`



`python run_ner_patent_clean.py --data_dir=data/bio --bert_model=bert-base-cased --task_name=patent --max_seq_length=64 --do_train --num_train_epochs 5 --do_eval --warmup_proportion=0.1 --output_dir=out_`



`python run_ner_patent_clean.py --data_dir=data/ --bert_model=bert-base-cased --task_name=ner --output_dir=out_base --max_seq_length=128 --do_train --num_train_epochs 5 --do_eval --warmup_proportion=0.1`


# Inference

```python
from bert import Ner

model = Ner("out_base/")

output = model.predict("Steve went to Paris")

print(output)
'''
    [
        {
            "confidence": 0.9981840252876282,
            "tag": "B-PER",
            "word": "Steve"
        },
        {
            "confidence": 0.9998939037322998,
            "tag": "O",
            "word": "went"
        },
        {
            "confidence": 0.999891996383667,
            "tag": "O",
            "word": "to"
        },
        {
            "confidence": 0.9991968274116516,
            "tag": "B-LOC",
            "word": "Paris"
        }
    ]
'''
```

# Deploy REST-API
BERT NER model deployed as rest api
```bash
python api.py
```
API will be live at `0.0.0.0:8000` endpoint `predict`
#### cURL request
` curl -X POST http://0.0.0.0:8000/predict -H 'Content-Type: application/json' -d '{ "text": "Steve went to Paris" }'`

Output
```json
{
    "result": [
        {
            "confidence": 0.9981840252876282,
            "tag": "B-PER",
            "word": "Steve"
        },
        {
            "confidence": 0.9998939037322998,
            "tag": "O",
            "word": "went"
        },
        {
            "confidence": 0.999891996383667,
            "tag": "O",
            "word": "to"
        },
        {
            "confidence": 0.9991968274116516,
            "tag": "B-LOC",
            "word": "Paris"
        }
    ]
}
```
#### cURL 
![curl output image](/img/curl.png)
#### Postman
![postman output image](/img/postman.png)

### Tensorflow version

- https://github.com/kyzhouhzau/BERT-NER
