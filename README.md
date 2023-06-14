# anglicism_removing

This repository contains code for article "Parameter efficient methods for anglicism substitution". Our work consists from two parts: dataset and code. 

## Ru_anglicism dataset

* [Dataset link](https://huggingface.co/datasets/shershen/ru_anglicism)

Dataset for detection and substraction anglicisms from sentences in Russian. Sentences with anglicism automatically parsed from National Corpus of the Russian language, Habr and Pikabu. The paraphrases for the sentences were created manually. Every row contains from 4 elements: anglicism - anglicism in the form in which it is used in the sentence - sentence with anglicism - paraphrased sentence with anglicism replaced. Example:

```
{
  'word': 'кринж',
  'form': 'кринжовую',
  'sentence': 'Моя подруга рассказала кринжовую историю.',
  'paraphrase': 'Моя подруга рассказала стыдную историю.'
}
```

This dataset can be used for substitution and detection tasks. In future we plan to expand dataset with new data

## Anglicism detection

Due to the small size of the dataset, we decided to focus on parameter efficient methods for solving anglicism detection and substitution problems.

In folder anglicism_detection you can find you can find notebooks in which we compare finetuning of normal small bert model and prefix tuning of big bert model. 
A model of even small sizes began to retrain from the second epoch, but a large model was trained for several epochs with a prefix-tuning approach, and with it we obtained a good result for the problem of detecting anglicisms. 

## Anglicism substitution 


In the folder anglicism_substitution you can find notebooks of different types of learning language models for paraphrasing sentences in which anglicism occurs. We tried different parameter efficient approaches, among which the best results were shown by prefix-tuning and learning lora weights. Also in this folder you can find an example of generating sentences using tensors trained during prefix-tuning.

