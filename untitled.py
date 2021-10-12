import pandas
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

ruta="/home/fabio/Documents/data_science_project_2/"
fichero="sample_linear_regression_data.txt"

datos=pandas.read_csv(ruta+fichero,header=None,sep=" ")

cols=list("Y")
cols.extend(["X"+str(i) for i in range(1,11)])
datos.columns=cols
print(datos)

lr = LinearRegression(regParam=0.0, #no regularization
                      fitIntercept=True,
                      standardization=False,
                      weightCol="weight" #no column weights (i.e. all instance weights=1.0)
                     )

lrevaluator = RegressionEvaluator(metricName='rmse')

lrmodel = lr.fit(datos)

print(evaluator.evaluate(datos))