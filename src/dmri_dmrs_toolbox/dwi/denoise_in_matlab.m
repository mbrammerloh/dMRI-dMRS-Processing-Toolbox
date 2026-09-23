function denoise_in_matlab(input,output,numvols, N, toolbox_path,type)
% Function that uses tensor MP-PCA implementation of matlab to denoise the
% dwi data containing multiple diffusion times. 
% Can denoise the data in 5D:
%       Data = [x, y, z, bvals/bvecs, diffTime]
% or 4D:
%       Data = [x, y, z, bvals/bvecs x diffTime]
%
% To be called from python script.
%
% INPUTS: 
%       input - string to path where the concatenated data (across diffusion times) is.
%       output - string to path where to save the denoised data. 
%       numvols - vector (1xM) with number of volumes for each diffusion
%       time M. Used to split the input 4D volume (x,y,z,M*bvals/bvecs) 
%       into a 5D matrix (x,y,z,bvals/bvecs,M) if needed
%       N  - patch/kernel size to use when denoising
%       toolbox_path - path to the folder where Matlab toolboxes are
%       type - string to the type of denoising one whishes to do (4D or 5D)
%       mask_path - string to path where the mask is
%
% OUTPUTS:
%       none
% 
% __________________________
% Rita Oliveira, 
% Mai 2025,
% MicMap Lab, Switzerland

    addpath(genpath(fullfile(toolbox_path,'spm12'))) 
    addpath(genpath(fullfile(toolbox_path,'Tensor-MP-PCA-main'))) 
    addpath(genpath(fullfile(toolbox_path,'MP-PCA-main'))) 
    disp('#########################')
    disp('>>> We are in matlab now')

    % Convert input strings to numbers
    N = str2num(N);
    numvols = str2num(numvols);

    if strcmp(type,'tMPPCA-5D')
        
        % Build 5D matrix with diffusion times in the 5th dimension
        dwi_total = spm_read_vols(spm_vol(input));
        for i=1:length(numvols)
            data_5D(:,:,:,:,i) = dwi_total(:,:,:,numvols(i)*(i-1)+1:numvols(i)*i);
        end

        % Denoised 5D matrix
        disp('#########################')
        disp('>>> Denoising data in 5D with tMPPCA, it might take a while...')
        [denoised_5D, sigma, P, SNR_gain] = denoise_recursive_tensor(data_5D,[N N N]);

        % Save denoised data
        disp('#########################')
        disp('>>> Saving denoised data')
        header = spm_vol(input);
        s = size(denoised_5D);  
        for i=1:s(4)*s(5)
             header(i).fname  = output;
             fprintf('saving volume %d \n',i)
             spm_write_vol(header(i),squeeze(denoised_5D(:,:,:,i)));
        end

    elseif strcmp(type,'tMPPCA-4D')

        % Build 4D matrix with diffusion times stacked at the 4th dimension
        data_4D = spm_read_vols(spm_vol(input));

        % Denoised 4D matrix
        disp('#########################')
        disp('>>> Denoising data in 4D with tMPPCA, it might take a while...')
        [denoised_4D, sigma, P, SNR_gain] = denoise_recursive_tensor(data_4D,[N N N]);

        % Save denoised data
        disp('#########################')
        disp('>>> Saving denoised data')
        header = spm_vol(input);
        s = size(denoised_4D);  
        for i=1:s(4)
             header(i).fname  = output;
             fprintf('saving volume %d \n',i)
             spm_write_vol(header(i),denoised_4D(:,:,:,i));
        end

    elseif strcmp(type,'MPPCA')

        % Build 4D matrix with diffusion times stacked at the 4th dimension
        data_4D = spm_read_vols(spm_vol(input));

        % Load mask if exists
        % if ~isempty(mask_path)
        %     mask = spm_read_vols(spm_vol(mask_path));
        %     fprintf('Using mask data: %s',mask_path)
        % else
        %     mask = [];
        % end

        % Denoised 4D matrix
        disp('#########################')
        disp('>>> Denoising data with MPPCA, it might take a while...')
        [denoised, sigma, P, SNR_gain] = denoise(data_4D,[N N N]);

        % Save denoised data
        disp('#########################')
        disp('>>> Saving denoised data')
        header = spm_vol(input);
        s = size(denoised);
        for i=1:s(4)
            header(i).fname  = output;
            fprintf('saving volume %d \n',i)
            spm_write_vol(header(i),denoised(:,:,:,i));
        end

    end

    % Save other stuff
    header_s = header(1);
    header_s.fname  = strrep(output, '.nii', '_SNR_gain.nii');
    spm_write_vol(header_s,SNR_gain);

    header_s = header(1);
    header_s.fname  = strrep(output, '.nii', '_sigma.nii');
    spm_write_vol(header_s,sqrt(sigma));

    header = header(1:size(P,4));
    for i=1:size(P,4)
        header(i).fname  = strrep(output, '.nii', '_P.nii');
        spm_write_vol(header(i),squeeze(P(:,:,:,i)));
    end


end
