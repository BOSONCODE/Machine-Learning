function [tr_labels,Av_train,te_labels,Av_test]=dataPro(A,labels,trainnumber)
 tr_labels=[];
    te_labels=[];
    Av_train=[];
    Av_test=[];
    ClassLabel = unique(labels);%标签具体类别
    nClass = length(ClassLabel);%标签类别数
    for c=1:nClass
         c_label = labels(labels==c);%提取标签为c的标签，依次存入标签c_label=[c,c,c……]Nc个c
         c_Data = A(:,labels==c);%提取标签为c的样本
         count(c) = size(c_Data,2);%记录类别为c的样本总数Nc
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
         data=randperm(count(c));%随机排列第c类的所有样本
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %——划分——训练集、测试集
         tr_labels=[tr_labels,c_label(data(1:trainnumber))];%选择trainnumber个样本的标签，作为训练样本标签
         Av_train=[Av_train,c_Data(:,data(1:trainnumber))];%选择trainnumber个样本，作为训练样本
         
         te_labels=[te_labels,c_label(data((trainnumber+1):count(c)))];%选择剩余样本的标签，作为测试样本标签
         Av_test=[Av_test,c_Data(:,data((trainnumber+1):count(c)))];%选择剩余样本，作为测试样本
    end