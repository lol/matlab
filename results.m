% Results and Discussion
clc;
clear all;

%subject_gdf{1} = 'ssvep-training-shiva-[2016.01.31-20.34.25].gdf';
%subject_gdf{2} = 'ssvep-record-train-[2016.04.09-10.38.44].gdf';
%subject_gdf{3} = 'ssvep-record-train-indra-3-[2016.03.31-23.42.46].gdf';


%subject_gdf{1} = 'ssvep-training-shiva-[2016.01.31-20.34.25].gdf';
%subject_gdf{2} = 'ssvep-record-train-[2016.04.09-10.38.44].gdf';
%subject_gdf{3} = 'ssvep-record-train-indra-3-[2016.03.31-23.42.46].gdf';


for i = 1:3
    %[confMat confMatK accuK] = ssvep_OnevsAll(subject_gdf{i});
    %[confMat confMatK accuK] = ssvep_native(subject_gdf{i});
    %[confMat confMatK accuK] = ssvep_OnevsAll_libsvm(subject_gdf{i});
    [confMat confMatK accuK] = ssvep_native_libsvm(subject_gdf{i});
    %fprintf('\n--- Resubstitution ---\n');
    %confMat
    fprintf('\n--- KFold Crossvalidation ---\n');
    confMatK
    percK = sum(diag(confMatK)) / sum(sum(confMatK)) * 100
    var(accuK)
    %box_accu(:, i) = accuK;
end

%subject = ['Subject 1'; 'Subject 2'; 'Subject 3';];
%boxplot(box_accu, subject); 