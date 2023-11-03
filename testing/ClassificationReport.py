from sklearn.metrics import classification_report
import pandas as pd

def ClassificationReport(test_data,test_labels,model):
    pred = model.predict(test_data)

    for ele in pred:
        for i in range(len(ele)):
            if ele[i] >= 0.5:
                ele[i] = 1
            else:
                ele[i] = 0
    report = pd.DataFrame(classification_report(test_data,pred,output_dict=True)).transpose()

    return report