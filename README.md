# covid19kaggle
The task here is after this kaggle challenge https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge/tasks
The aim is to identify relevant articles to answer different questions about covid.

Existing code analyzes the titles of all articles, identifies parts of speech and groups articles under similat topics using LDA.

# Some example outputs

### Most common title word is **virus**

### Nouns in transmission articles titles
nouns_doc[0:10]

[['feline', 'peritonitis', 'transmission'],

 ['abstraction', 'disease', 'marmoset', 'survey', 'transmission'],
 
 ['natural', 'plant', 'protein', 'transmission', 'vector', 'virus'],
 
 ['disease', 'network', 'transmission'],
 
 ['chain', 'chapter', 'food', 'path', 'transmission'],
 
 ['chapter', 'epidemiology', 'transmission', 'virus'],
 
 ['focus', 'infection', 'overview', 'prevention', 'transmission'],
 
 ['focus', 'infection', 'overview', 'prevention', 'transmission'],
 
 ['control',
  'diagnosis',
  'disease',
  'prevention',
  'principle',
  'transmission'],
 
 ['disease', 'network', 'transmission']]

### Organized as a bag of words for every article
corpus[0:10]

[[(0, 1), (1, 1), (2, 1)],

[(2, 1), (3, 1), (4, 1), (5, 1), (6, 1)],

[(2, 1), (7, 1), (8, 1), (9, 1), (10, 1), (11, 1)],

[(2, 1), (4, 1), (12, 1)],

[(2, 1), (13, 1), (14, 1), (15, 1), (16, 1)],

[(2, 1), (11, 1), (14, 1), (17, 1)],

[(2, 1), (18, 1), (19, 1), (20, 1), (21, 1)],

[(2, 1), (18, 1), (19, 1), (20, 1), (21, 1)],

[(2, 1), (4, 1), (21, 1), (22, 1), (23, 1), (24, 1)],

[(2, 1), (4, 1), (12, 1)]]

### Get 20 topics out of all transmission article titles

lda_model.print_topics()

[(0,
  '0.145*"transmission" + 0.053*"hazard" + 0.031*"outbreak" + 0.027*"virus" + 0.022*"factor" + 0.018*"data" + 0.018*"reappraisal" + 0.015*"approach" + 0.013*"care" + 0.013*"disease"'),

(1,
  '0.151*"transmission" + 0.035*"pathogen" + 0.032*"virus" + 0.030*"family" + 0.029*"survey" + 0.027*"genome" + 0.027*"sequence" + 0.025*"moral_force" + 0.025*"China" + 0.024*"data"'),

(2,
  '0.157*"transmission" + 0.032*"virus" + 0.026*"form" + 0.022*"disease" + 0.018*"Thailand" + 0.018*"chapter" + 0.018*"temperature" + 0.016*"Wuhan" + 0.015*"function" + 0.015*"droplet"'),

(3,
  '0.123*"transmission" + 0.039*"disease" + 0.021*"potential" + 0.020*"center" + 0.017*"east" + 0.016*"outbreak" + 0.014*"syndrome" + 0.014*"evidence" + 0.013*"novel" + 0.012*"sociable"'),

(4,
  '0.147*"transmission" + 0.030*"virus" + 0.025*"disease" + 0.022*"case" + 0.018*"homo" + 0.017*"Korea" + 0.017*"function" + 0.016*"infection" + 0.014*"moral_force" + 0.013*"survey"'),

(5,
  '0.150*"transmission" + 0.044*"disease" + 0.023*"network" + 0.022*"infection" + 0.021*"homo" + 0.020*"virus" + 0.020*"epidemic" + 0.018*"model" + 0.018*"rate" + 0.016*"influenza"'),

(6,
  '0.146*"transmission" + 0.027*"influenza" + 0.025*"moral_force" + 0.022*"pathogen" + 0.021*"pandemic" + 0.019*"mold" + 0.019*"China" + 0.018*"disease" + 0.017*"outbreak" + 0.015*"function"'),

(7,
  '0.137*"transmission" + 0.056*"center" + 0.052*"east" + 0.051*"syndrome" + 0.030*"infection" + 0.018*"virus" + 0.016*"worker" + 0.014*"potential" + 0.014*"case" + 0.013*"lack"'),

(8,
  '0.149*"transmission" + 0.033*"disease" + 0.025*"patient" + 0.023*"feature" + 0.019*"model" + 0.018*"setting" + 0.018*"general" + 0.018*"evidence" + 0.017*"hazard" + 0.017*"commercial"'),

(9,
  '0.141*"transmission" + 0.053*"control" + 0.047*"virus" + 0.040*"China" + 0.028*"consequence" + 0.024*"network" + 0.021*"population" + 0.020*"epidemic" + 0.019*"disease" + 0.016*"vector"'),

(10,
  '0.153*"transmission" + 0.064*"model" + 0.041*"disease" + 0.031*"novel" + 0.024*"data" + 0.017*"outbreak" + 0.016*"hospital" + 0.015*"influenza" + 0.014*"Wuhan" + 0.012*"state"'),

(11,
  '0.167*"transmission" + 0.040*"severe_acute_respiratory_syndrome" + 0.033*"control" + 0.031*"moral_force" + 0.030*"infection" + 0.028*"survey" + 0.022*"virus" + 0.019*"China" + 0.019*"healthcare" + 0.016*"modeling"'),

(12,
  '0.113*"transmission" + 0.049*"impact" + 0.045*"appraisal" + 0.035*"moral_force" + 0.029*"building" + 0.025*"populace" + 0.025*"health" + 0.021*"flat" + 0.021*"tracer" + 0.021*"gas"'),

(13,
  '0.141*"transmission" + 0.107*"novel" + 0.063*"Wuhan" + 0.044*"outbreak" + 0.042*"virus" + 0.041*"development" + 0.035*"surveillance" + 0.025*"macaque" + 0.021*"horizontal" + 0.019*"pathogen"'),

(14,
  '0.165*"transmission" + 0.134*"influenza" + 0.072*"pandemic" + 0.070*"virus" + 0.046*"state" + 0.036*"potential" + 0.028*"manner" + 0.024*"China" + 0.021*"reappraisal" + 0.018*"United_States"'),

(15,
  '0.156*"transmission" + 0.028*"China" + 0.027*"disease" + 0.020*"Vietnam" + 0.018*"homo" + 0.017*"beginning" + 0.016*"behavior" + 0.016*"potential" + 0.016*"blood" + 0.016*"chain"'),

(16,
  '0.136*"transmission" + 0.047*"case" + 0.035*"epidemic" + 0.027*"moral_force" + 0.027*"intervention" + 0.026*"Korea" + 0.025*"exploitation" + 0.017*"South" + 0.017*"China" + 0.017*"disease"'),

(17,
  '0.136*"transmission" + 0.059*"disease" + 0.029*"virus" + 0.023*"hazard" + 0.022*"network" + 0.015*"prediction" + 0.015*"patient" + 0.012*"model" + 0.012*"influenza" + 0.012*"contact"'),

(18,
  '0.146*"transmission" + 0.033*"infection" + 0.031*"influenza" + 0.027*"virus" + 0.021*"disease" + 0.019*"city" + 0.018*"homo" + 0.017*"novel" + 0.016*"outbreak" + 0.015*"family"'),

(19,
  '0.138*"transmission" + 0.055*"virus" + 0.026*"acute_accent" + 0.024*"infection" + 0.023*"influenza" + 0.021*"syndrome" + 0.020*"epidemic" + 0.019*"survey" + 0.018*"hazard" + 0.017*"contact"')]

### Get topic distribution for a particular article title

lda_model.get_document_topics(corpus[0])

[(15, 0.940625)]

lda_model.get_document_topics(corpus[1])

[(11, 0.9634615)]

lda_model.get_document_topics(corpus[10])

[(2, 0.48466748), (12, 0.48630023)]
