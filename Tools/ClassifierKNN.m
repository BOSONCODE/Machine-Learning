function [ Acc_KNN ] = ClassifierKNN( TraData,tra_labels,Tes,tes_labels)
train_data=TraData';
train_label=tra_labels';
test_data=Tes';
test_label=tes_labels';
mdl = ClassificationKNN.fit(train_data,train_label,'NumNeighbors',1);
predict_label=predict(mdl, test_data);
Acc_KNN=length(find(predict_label == test_label))/length(test_label);
end

