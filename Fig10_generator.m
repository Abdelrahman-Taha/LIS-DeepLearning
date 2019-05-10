%clearvars
%close all
%clc

%% Description:
%
% This is the main code for generating Figure 10 in the original article
% mentioned below.
%
% version 1.0 (Last edited: 2019-05-10)
%
% The definitions and equations used in this code refer (mostly) to the 
% following publication:
%
% Abdelrahman Taha, Muhammad Alrabeiah, and Ahmed Alkhateeb, "Enabling 
% Large Intelligent Surfaces with Compressive Sensing and Deep Learning," 
% arXiv e-prints, p. arXiv:1904.10136, Apr 2019. 
% [Online]. Available: https://arxiv.org/abs/1904.10136
%
% The DeepMIMO dataset is adopted.  
% [Online]. Available: http://deepmimo.net/
%
% License: This code is licensed under the GPLv3 license. If you in any way
% use this code for research that results in publications, please cite our
% original article mentioned above.

%% System Model parameters

kbeams=1;   %select the top kbeams, get their feedback and find the max actual achievable rate 
Pt=5; % dB
L =1; % number of channel paths (L)
% Note: The axes of the antennas match the axes of the ray-tracing scenario
My_ar=[32 64]; % number of LIS reflecting elements across the y axis
Mz_ar=[32 64]; % number of LIS reflecting elements across the z axis
M_bar=8; % number of active elements
K_DL=64; % number of subcarriers as input to the Deep Learning model
Training_Size=[2  1e4*(1:.4:3)]; % Training Dataset Size vector

% Preallocation of output variables
Rate_DLt=zeros(numel(My_ar),numel(Training_Size)); 
Rate_OPTt=zeros(numel(My_ar),numel(Training_Size));

%% Figure Data Generation 

for rr = 1:1:numel(My_ar)
    save Fig10_data.mat L My_ar Mz_ar M_bar Training_Size K_DL Rate_DLt Rate_OPTt
    [Rate_DL,Rate_OPT]=Main_fn(L,My_ar(rr),Mz_ar(rr),M_bar,K_DL,Pt,kbeams,Training_Size);
    Rate_DLt(rr,:)=Rate_DL; Rate_OPTt(rr,:)=Rate_OPT;
end

save Fig10_data.mat L My_ar Mz_ar M_bar Training_Size K_DL Rate_DLt Rate_OPTt 

%% Figure Plot

%------------- Figure Input Variables ---------------------------%
% M; My_ar; Mz_ar; M_bar; 
% Training_Size; Rate_DLt; Rate_OPTt;

%------------------ Fixed Parameters ----------------------------%
% Full Regression 
% L = min = 1
% K = 512, K_DL = max = 64
% M_bar = 8
% random distribution of active elements

Colour = 'brgmcky';

f10 = figure('Name', 'Figure10', 'units','pixels');
hold on; grid on; box on;
title(['Achievable Rate for different dataset sizes using only ' num2str(M_bar) ' active elements'],'fontsize',12)
xlabel('Deep Learning Training Dataset Size (Thousands of Samples)','fontsize',14)
ylabel('Achievable Rate (bps/Hz)','fontsize',14)
set(gca,'FontSize',13)
if ishandle(f10)
    set(0, 'CurrentFigure', f10)
    hold on; grid on;
    for rr=1:1:numel(My_ar)
        plot((Training_Size*1e-3),Rate_OPTt(rr,:),[Colour(rr) '*--'],'markersize',8,'linewidth',2, 'DisplayName',['Genie-Aided Reflection Beamforming, M = ' num2str(My_ar(rr)) '*' num2str(Mz_ar(rr))])
        plot((Training_Size*1e-3),Rate_DLt(rr,:),[Colour(rr) 's-'],'markersize',8,'linewidth',2, 'DisplayName', ['DL Reflection Beamforming, M = ' num2str(My_ar(rr)) '*' num2str(Mz_ar(rr))])
    end
    legend('Location','SouthEast')
    legend show
end
drawnow
hold off