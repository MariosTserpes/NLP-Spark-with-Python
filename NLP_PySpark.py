from pyspark.sql import SparkSession
from pyspark.sql.functions import length

from pyspark.ml import Pipeline

from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, IDF, StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import NaiveBayes

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

spark = SparkSession.builder.appName('NLP').getOrCreate()

data = spark.read.csv('SMSSpamCollection', 
                      inferSchema = True,
                      sep = '\t')
data = data.withColumnRenamed('_c0', 'class').withColumnRenamed('_c1', 'text')
data.show()

#Cleaning and Preparing the Data.
data = data.withColumn('length', length(data['text']))
data.show()

'''
There is any difference between the length of a ham vs spam column?
'''
data.groupBy('class').mean().show()

'''
Text Mining Techniques for NLP
'''
tokenizer = Tokenizer(inputCol = 'text', outputCol = 'token_text')
stop_remove = StopWordsRemover(inputCol = 'token_text', outputCol = 'stop_token')
count_vect = CountVectorizer(inputCol = 'stop_token', outputCol = 'c_vec')
IDFs = IDF(inputCol = 'c_vec', outputCol = 'tf_idf')
ham_spam_to_numeric = StringIndexer(inputCol = 'class', outputCol = 'label')

'''
VectorAssembler
'''
clean_up = VectorAssembler(inputCols = ['tf_idf', 'length'], outputCol = 'features')

'''
NaiveBayes
'''
NBc = NaiveBayes()

'''
NLP Pipeline
'''
data_prep_pipeline = Pipeline(stages = [ham_spam_to_numeric,
                                        tokenizer,
                                        stop_remove,
                                        count_vect,
                                        IDFs,
                                        clean_up])

cleaner = data_prep_pipeline.fit(data)
clean_data = cleaner.transform(data)
clean_data = clean_data.select('label', 'features')
clean_data.show()

'''
Splitting train and test
'''
training, test = clean_data.randomSplit([0.7, 0.3])

spam_detector = NBc.fit(training)
data.printSchema()

test_results = spam_detector.transform(test)
test_results.show()

'''
Evaluation
'''
accuracy_eval = MulticlassClassificationEvaluator()
accuracy = accuracy_eval.evaluate(test_results)

print(f"Accuracy of NaiveBayes Model {accuracy}")