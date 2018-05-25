function [tr_labels,Av_train,te_labels,Av_test]=dataPro(A,labels,trainnumber)
 tr_labels=[];
    te_labels=[];
    Av_train=[];
    Av_test=[];
    ClassLabel = unique(labels);%��ǩ�������
    nClass = length(ClassLabel);%��ǩ�����
    for c=1:nClass
         c_label = labels(labels==c);%��ȡ��ǩΪc�ı�ǩ�����δ����ǩc_label=[c,c,c����]Nc��c
         c_Data = A(:,labels==c);%��ȡ��ǩΪc������
         count(c) = size(c_Data,2);%��¼���Ϊc����������Nc
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
         data=randperm(count(c));%������е�c�����������
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %�������֡���ѵ���������Լ�
         tr_labels=[tr_labels,c_label(data(1:trainnumber))];%ѡ��trainnumber�������ı�ǩ����Ϊѵ��������ǩ
         Av_train=[Av_train,c_Data(:,data(1:trainnumber))];%ѡ��trainnumber����������Ϊѵ������
         
         te_labels=[te_labels,c_label(data((trainnumber+1):count(c)))];%ѡ��ʣ�������ı�ǩ����Ϊ����������ǩ
         Av_test=[Av_test,c_Data(:,data((trainnumber+1):count(c)))];%ѡ��ʣ����������Ϊ��������
    end