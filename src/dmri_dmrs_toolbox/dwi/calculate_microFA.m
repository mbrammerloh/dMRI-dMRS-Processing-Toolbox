function calculate_microFA(LTE_path,STE_path, header_path, output_path, toolbox_path)
% Function that uses tensor MP-PCA implementation of matlab to denoise the
% dwi data containing multiple diffusion times
% Data = [x, y, z, bvals/bvecs, diffTime]
% To ba called from python script
%
% INPUTS: 
%       LTE_path - string of the path to the input LTE nifti file 
%       STE_path - string of the path to the input STE nifti file 
%       output_path - string to the output folder where to save the results
%       toolbox_path - path to the folder where Matlab toolboxes are
%
% OUTPUTS:
%       none
% 
% __________________________
% Rita Oliveira, 
% Mai 2025,
% MicMap Lab, Switzerland

    addpath(genpath(fullfile(toolbox_path,'spm12'))) 
    addpath(genpath(fullfile(toolbox_path,'md-dmri-master'))) 
    disp('#########################')
    disp('>>> We are in matlab now')

    % Define path to nii file 
    s{1}.nii_fn = LTE_path;
    s{2}.nii_fn = STE_path;
    disp(LTE_path)
    disp(STE_path)

    % Firt encoding is linear and the second is spherical
    b_deltas  = [1 0];

    cd(output_path)
    merged_nii_name = 'LTE_and_STE';
    disp(merged_nii_name)

    % Loop over nii files to create partial xps structures, and store them in the cell array.
    for i = 1:2
        [bval_fn, bvec_fn] = mdm_fn_nii2bvalbvec(s{i}.nii_fn);
        s{i}.xps     = mdm_xps_from_bval_bvec(bval_fn, bvec_fn, b_deltas(i));
        % b = s{i}.xps.b;
        % u = [1 0 0];
        % b_radial = b * (1 - b_deltas(i)) / 3;
        % b_axial  = b - 2 * b_radial;
        % if i==2
        %     s{i}.xps.bt = xps.bt;
        % else
        %     s{i}.xps.bt = tm_1x3_to_1x6(b_axial, b_radial, u);
        % end

    end

    % Merge the s structure, and save the merged nii along with its corresponding xps.mat file.
    s_merged = mdm_s_merge(s, output_path, merged_nii_name);

    % Podwer average
    disp('>>> Doing powder average. It takes time...')
    opt = mdm_opt;
    opt.do_overwrite = 1;
    opt.verbose      = 1;
    s_pa = mdm_s_powder_average(s_merged, output_path, opt);
   
    % Fit the data with gamma
    disp('>>> Fitting data. It takes time...')
    opt=dtd_gamma_opt();
    mfs_fn = dtd_gamma_4d_data2fit(s_pa, fullfile(output_path,'result.mat'), opt);
    
    % Fit the data with covariance (not working well)
    %opt=dtd_covariance_opt;
    %dtd_covariance_pipe(s_merged, fullfile(output_path,'result.mat'), opt);

    % Compute FA
    disp('>>> Computing microFA...')
    load(fullfile(output_path,'result.mat'))
    m=mfs.m;
    S0 = m(:,:,:,1);
    MD = m(:,:,:,2) / 1e-9;
    MKi = 3*m(:,:,:,3)./(m(:,:,:,2).^2);
    MKa = 3*m(:,:,:,4)./(m(:,:,:,2).^2);
    Uaniso = m(:,:,:,4) / (1e-9)^2; 
    Uiso   = m(:,:,:,3) / (1e-9)^2;
    Vl = 5/2 * m(:,:,:,4);
    uFA = sqrt(3/2) * sqrt( Vl ./ (Vl + m(:,:,:,3) + m(:,:,:,2).^2) );

    % delta_u = (m(:,:,:,4) -m(:,:,:,3))./(m(:,:,:,2).^2);
    % delta_u_safe = delta_u;
    % delta_u_safe(abs(delta_u) < 1e-6) = 1e-6; % Prevent instability
    % uFA2 = sqrt(3/2) * (1 + 2/5 * 1./delta_u_safe).^(-1/2);
    % 

    header = spm_vol(header_path);

    header.fname  = fullfile(output_path,'microFA.nii');
    header.dt = [16 0];
    spm_write_vol(header,uFA);

    header.fname  = fullfile(output_path,'MD.nii');
    header.dt = [16 0];
    spm_write_vol(header,MD);

    header.fname  = fullfile(output_path,'Uaniso.nii');
    header.dt = [16 0];
    spm_write_vol(header,Uaniso);

end
