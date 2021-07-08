% Script to create network maps for the Glasser parcellations
%   requires NIfTI toolbox to and load and save .nii images
%   presumes Glasser parcellation is in the current dictory
% addpath(genpath('/Users/andrew/Documents/MATLAB/NIfTI_20140122'));

% Load Glasser parcellation
wholeMap = load_untouch_nii('MMP_in_MNI_corr.nii');
% Renumber to be continuous from 1 to 360 (right half is 1-180, left half
% is 201-380 in atlas)
for i = 201:380
    wholeMap.img(wholeMap.img == i) = i-20;
end
% node to network dictionary
nodeNets = {'Visual1' 'Visual2' 'Visual2' 'Visual2' 'Visual2' 'Visual2' 'Visual2','Somatomotor' 'Somatomotor' 'Cingulo-Oper' 'Language' 'Default' 'Visual2','Frontopariet' 'Frontopariet' 'Visual2' 'Visual2' 'Visual2' 'Visual2','Visual2' 'Visual2' 'Visual2' 'Visual2' 'Auditory' 'Default' 'Default','Dorsal-atten' 'Default' 'Frontopariet' 'Posterior-Mu' 'Posterior-Mu','Posterior-Mu' 'Posterior-Mu' 'Posterior-Mu' 'Posterior-Mu' 'Somatomotor','Cingulo-Oper' 'Cingulo-Oper' 'Somatomotor' 'Somatomotor' 'Somatomotor','Somatomotor' 'Cingulo-Oper' 'Cingulo-Oper' 'Cingulo-Oper' 'Language','Somatomotor' 'Visual2' 'Visual2' 'Language' 'Somatomotor' 'Somatomotor','Somatomotor' 'Somatomotor' 'Somatomotor' 'Somatomotor' 'Cingulo-Oper','Cingulo-Oper' 'Cingulo-Oper' 'Cingulo-Oper' 'Posterior-Mu','Posterior-Mu' 'Frontopariet' 'Posterior-Mu' 'Posterior-Mu','Posterior-Mu' 'Posterior-Mu' 'Posterior-Mu' 'Posterior-Mu','Posterior-Mu' 'Posterior-Mu' 'Posterior-Mu' 'Frontopariet' 'Default','Default' 'Posterior-Mu' 'Frontopariet' 'Cingulo-Oper' 'Default','Frontopariet' 'Default' 'Frontopariet' 'Frontopariet' 'Cingulo-Oper','Frontopariet' 'Cingulo-Oper' 'Posterior-Mu' 'Posterior-Mu','Frontopariet' 'Posterior-Mu' 'Frontopariet' 'Frontopariet','Posterior-Mu' 'Posterior-Mu' 'Language' 'Language' 'Frontopariet','Frontopariet' 'Cingulo-Oper' 'Somatomotor' 'Somatomotor' 'Somatomotor','Auditory' 'Auditory' 'Cingulo-Oper' 'Cingulo-Oper' 'Auditory','Cingulo-Oper' 'Cingulo-Oper' 'Orbito-Affec' 'Frontopariet','Orbito-Affec' 'Cingulo-Oper' 'Cingulo-Oper' 'Somatomotor' 'Language','Language' 'Posterior-Mu' 'Posterior-Mu' 'Posterior-Mu' 'Visual1','Ventral-Mult' 'Default' 'Auditory' 'Default' 'Posterior-Mu' 'Language','Default' 'Default' 'Posterior-Mu' 'Posterior-Mu' 'Posterior-Mu','Frontopariet' 'Posterior-Mu' 'Ventral-Mult' 'Language' 'Language','Visual2' 'Default' 'Dorsal-atten' 'Dorsal-atten' 'Visual1' 'Language','Frontopariet' 'Frontopariet' 'Language' 'Cingulo-Oper' 'Cingulo-Oper','Frontopariet' 'Posterior-Mu' 'Posterior-Mu' 'Visual2' 'Visual2','Visual2' 'Posterior-Mu' 'Visual2' 'Visual2' 'Visual2' 'Visual2','Visual2' 'Posterior-Mu' 'Posterior-Mu' 'Visual2' 'Posterior-Mu','Posterior-Mu' 'Orbito-Affec' 'Cingulo-Oper' 'Somatomotor' 'Cingulo-Oper','Frontopariet' 'Frontopariet' 'Default' 'Auditory' 'Auditory' 'Auditory','Posterior-Mu' 'Posterior-Mu' 'Cingulo-Oper' 'Cingulo-Oper','Cingulo-Oper' 'Visual1' 'Visual2' 'Visual2' 'Visual2' 'Visual2','Visual2' 'Visual2' 'Somatomotor' 'Somatomotor' 'Cingulo-Oper','Cingulo-Oper' 'Default' 'Visual2' 'Frontopariet' 'Frontopariet','Visual2' 'Visual2' 'Visual2' 'Visual2' 'Visual2' 'Visual2' 'Visual2','Visual2' 'Auditory' 'Cingulo-Oper' 'Default' 'Dorsal-atten','Dorsal-atten' 'Frontopariet' 'Posterior-Mu' 'Posterior-Mu','Posterior-Mu' 'Posterior-Mu' 'Posterior-Mu' 'Posterior-Mu' 'Somatomotor','Cingulo-Oper' 'Cingulo-Oper' 'Somatomotor' 'Somatomotor' 'Somatomotor','Somatomotor' 'Cingulo-Oper' 'Cingulo-Oper' 'Cingulo-Oper' 'Language','Somatomotor' 'Visual2' 'Visual2' 'Language' 'Somatomotor' 'Somatomotor','Somatomotor' 'Somatomotor' 'Somatomotor' 'Somatomotor' 'Cingulo-Oper','Frontopariet' 'Cingulo-Oper' 'Cingulo-Oper' 'Posterior-Mu','Frontopariet' 'Frontopariet' 'Posterior-Mu' 'Posterior-Mu','Posterior-Mu' 'Posterior-Mu' 'Posterior-Mu' 'Posterior-Mu','Posterior-Mu' 'Posterior-Mu' 'Posterior-Mu' 'Frontopariet','Frontopariet' 'Default' 'Posterior-Mu' 'Frontopariet' 'Cingulo-Oper','Default' 'Frontopariet' 'Frontopariet' 'Cingulo-Oper' 'Frontopariet','Cingulo-Oper' 'Frontopariet' 'Cingulo-Oper' 'Posterior-Mu','Posterior-Mu' 'Frontopariet' 'Posterior-Mu' 'Frontopariet','Frontopariet' 'Frontopariet' 'Posterior-Mu' 'Language' 'Language','Frontopariet' 'Frontopariet' 'Cingulo-Oper' 'Somatomotor' 'Somatomotor','Somatomotor' 'Auditory' 'Somatomotor' 'Cingulo-Oper' 'Cingulo-Oper','Auditory' 'Cingulo-Oper' 'Cingulo-Oper' 'Orbito-Affec' 'Frontopariet','Orbito-Affec' 'Cingulo-Oper' 'Cingulo-Oper' 'Somatomotor' 'Language','Language' 'Posterior-Mu' 'Posterior-Mu' 'Posterior-Mu' 'Visual1','Ventral-Mult' 'Default' 'Auditory' 'Default' 'Posterior-Mu' 'Language','Posterior-Mu' 'Default' 'Posterior-Mu' 'Posterior-Mu' 'Posterior-Mu','Frontopariet' 'Posterior-Mu' 'Ventral-Mult' 'Language' 'Language','Visual2' 'Default' 'Dorsal-atten' 'Dorsal-atten' 'Visual1' 'Language','Frontopariet' 'Frontopariet' 'Language' 'Cingulo-Oper' 'Cingulo-Oper','Frontopariet' 'Posterior-Mu' 'Posterior-Mu' 'Visual2' 'Visual2','Visual2' 'Posterior-Mu' 'Visual2' 'Visual2' 'Visual2' 'Visual2','Visual2' 'Posterior-Mu' 'Frontopariet' 'Visual2' 'Posterior-Mu','Posterior-Mu' 'Orbito-Affec' 'Cingulo-Oper' 'Somatomotor' 'Cingulo-Oper','Frontopariet' 'Frontopariet' 'Default' 'Auditory' 'Auditory' 'Auditory','Posterior-Mu' 'Frontopariet' 'Cingulo-Oper' 'Cingulo-Oper','Cingulo-Oper'};
% list of networks
nets = {'Auditory','Cingulo-Oper','Default','Dorsal-atten','Frontopariet','Language','Orbito-Affec','Posterior-Mu','Somatomotor','Ventral-Mult','Visual1','Visual2'};

% generate individual network maps
for i = 1:length(nets)
    % copy map from atlas, zero it out
    netMap = wholeMap;
    netMap.img(netMap.img ~= 0) = 0;
    for j = 1:length(nodeNets)
        if strcmp(nodeNets{j}, nets{i})
            netMap.img(wholeMap.img == j) = 1;
        else
            netMap.img(wholeMap.img == j) = 0;
        end
    end
    save_untouch_nii(netMap,nets{i});
end
