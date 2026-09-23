import os
import re
import shutil
import numpy as np
import nibabel as nib
os.environ.pop("MPLBACKEND", None)
import matplotlib.pyplot as plt
import sys
import tkinter as tk
from tkinter import ttk
from scipy.ndimage import label, find_objects
import subprocess
import json
import math
from itertools import groupby
import gzip
import tempfile
import glob as glob
import fnmatch
from .atlas_functions import create_ROI_mask

##### FILES AND SYSTEM OPERATIONS #####

def gunzip_file(input_path):
    output_path = input_path.replace('.nii.gz','.nii')  # removes the .gz extension

    with gzip.open(input_path, 'rb') as f_in:
        with open(output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    return output_path

def gzip_file(input_path):
    output_path = input_path.replace('.nii','.nii.gz')  # removes the .gz extension

    with open(input_path, 'rb') as f_in:
        with gzip.open(output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    return output_path

def generate_paths(main_path, list_scanNo):

    list_paths = []
    for items in list_scanNo:
        list_paths.append(os.path.join(main_path, str(items)))
    return (dict(zip(list_scanNo, list_paths)))


# def gen_bids_path(suffix):

#     if bids_strc.get("description"):
#         description = bids_strc["description"]
#         base_name = f'{bids_strc["subject"]}_{
#             bids_strc["session"]}_{description}_'
#     else:
#         description = ""  # Default to an empty string if description is None or empty
#         base_name = f'{bids_strc["subject"]}_{bids_strc["session"]}_'

#     out_path = os.path.join(bids_strc["root"],
#                             bids_strc["subject"],
#                             bids_strc["session"],
#                             bids_strc["datatype"],
#                             *(description,) if description else (),
#                             base_name + suffix)
#     return out_path


def extract_paths(dict_paths, list_scans):

    extracted_paths = []
    for key in list_scans:
        if key != 0:
            extracted_paths.append(dict_paths.get(key))
    return extracted_paths


def show_waiting_window(message):

    # Create the root window
    root = tk.Tk()
    root.title(message)

    # Set the size of the window
    root.geometry("300x100")

    # Add a label to the window
    label = ttk.Label(root, text="Please wait...")
    label.pack(pady=10)

    # Add an OK button to the window
    ok_button = ttk.Button(root, text="OK", command=root.destroy)
    ok_button.pack(pady=10)

    # This makes the window appear in the center of the screen
    root.eval('tk::PlaceWindow . center')

    # Run the application
    root.mainloop()


def replace_string(input_list_string, string_to_replace, replacing_string):

    output_list_string = []
    for ii in range(len(input_list_string)):
        output_list_string.append(input_list_string[ii].replace(
            string_to_replace, replacing_string))
    return output_list_string


# def create_directory(path):

#     if not os.path.exists(path):
#         os.makedirs(path)

def create_directory(directory_name):
     try:
         if not os.path.exists(directory_name):
            os.makedirs(directory_name)
     except FileExistsError:
         print(f"Warn: Directory '{directory_name}' already exists.")
     except PermissionError:
         print(f"Err: Permission denied: Unable to create '{directory_name}'.")
     except Exception as e:
        print(f"An error occurred: {e}")


def delete_directory_sudo(path):
    
    call = [f' sudo rm -r {path}']
    os.system(' '.join(call))
    
def copy_file(source_paths, destination_paths):

    for source, destination in zip(source_paths, destination_paths):
        shutil.copyfile(source, destination)

def copy_files_BIDS(bids_strc, output_path, filename):

     out_f  = os.path.join(output_path, filename)
     in_f   =  get_file_in_folder(bids_strc, '*'+filename)
     copy_files([in_f],[out_f])
    
     # new_path = os.path.join(output_path,bids_strc.get_param("base_name")+filename)
     # copy_file([bids_strc.get_path(filename)], 
     #           [new_path])
     
     return out_f

def get_file_in_folder(bids_strc, pattern):
   try:
       return glob.glob(os.path.join(bids_strc.get_path(), pattern))[0]
   except (IndexError, FileNotFoundError):
       return []
    
def get_subdir(directory_path):
    subdirectories = [name for name in os.listdir(
        directory_path) if os.path.isdir(os.path.join(directory_path, name))]
    if len(subdirectories) > 0:
        return os.path.join(directory_path, subdirectories[0])
    return None


def get_filename(file_path):

    file_name = os.path.basename(file_path)
    file_name_no_extension = file_name.split('.')[0]

    return file_name_no_extension


def get_files_with_extension(folder_path, extension):
    files = []
    for file_name in os.listdir(folder_path):
        # Create the full path to the file
        file_path = os.path.join(folder_path, file_name)
        # Check if the path points to a file and has the desired extension
        if os.path.isfile(file_path) and file_name.endswith(extension):
            files.append(file_path)
    return files


def remove_file(input_path):

    call = [f'rm',
            f'{input_path}']

    os.system(' '.join(call))


def concat_files(list_files, output_path):

    with open(output_path, 'w') as fout:
        for lines in zip(*[open(fname).readlines() for fname in list_files]):
            lines = [k.rstrip('\n') for k in lines]
            fout.write(' '.join(lines) + '\n')

def unconcat_files(file_path, delta_path):
    from collections import defaultdict
    
    # Load and clean delta values
    deltas = [int(float(x)) for x in read_numeric_txt(delta_path)[0]]
    
    # Load and clean file values
    with open(file_path, 'r') as f:
        values = [[float(x) for x in line.strip().split()] for line in f if line.strip()]
    
    # Group indices by delta
    delta_groups = defaultdict(list)
    for idx, delta in enumerate(deltas):
        delta_groups[delta].append(idx)

    # Save each group to a new NIfTI file
    for delta, indices in delta_groups.items():

        delta_folder = os.path.join(os.path.join(os.path.dirname(file_path)), f'Delta_{delta}')
        os.makedirs(delta_folder, exist_ok=True)
        
        name = os.path.basename(file_path).replace('allDelta-allb', f'Delta_{delta}')
        output_path = os.path.join(delta_folder, f'{name}')
        if '.eddy_rotated_bvecs' in output_path:
            #output_path = output_path.replace('.eddy_rotated_bvecs', '_bvecs_rotated.txt')
            output_path = re.sub(r'(Delta_\d+_).*$', r'\1bvecsRotated.txt', output_path)
            
        extracted_lines = []
        for line in values:
            new_values = [line[i] for i in indices]
            extracted_lines.append(' '.join(map(str, new_values)))

        # Write to output
        with open(output_path, 'w') as f:
            f.write('\n'.join(extracted_lines))
        print(f"Saved: {output_path}")
        
def write_txt(values, output_path, mode):
    ''' mode is 'w' for write and 'a' for append '''
    with open(output_path, mode) as f:
        f.write(' '.join(map(str, values)))
    f.close()


def copy_files(source_paths, destination_paths):

    for source, destination in zip(source_paths, destination_paths):
        shutil.copyfile(source, destination)


def read_numeric_txt(input_path):

    with open(input_path, 'r') as file:
        data = file.read()
        values = [float(num) for num in data.split()]

    values = np.array(values)[np.newaxis, :]
    return values


def remove_folder(folder_path):

    call = [f'rm -rf {folder_path}']
    os.system(' '.join(call))

def update_cfg(cfg):
    
    ordered_steps = [
        'redo_all',
        'redo_bet_anat',
        'redo_b0_extract',
        'redo_merge_dwi',
        'redo_firstmask',
        'redo_denoise',
        'redo_gibbs',
        'redo_topup',
        'redo_eddy',
        'redo_final_mask']

    update_flag = False
    for step in ordered_steps:
        if cfg[step] == 1:
            update_flag = True
        if update_flag:
            cfg[step] = 1

    with open(cfg['data_path'] + '/.config.json', 'w') as f:
        json.dump(cfg, f)

    return cfg

def create_composite_page(images, titles=None, heading=None, subtext=None, page_size=(1240, 1754), margin=60,bottom_margin=180):
    from PIL import Image, ImageDraw, ImageFont
    """Compose a page with up to 2 images and an optional heading and individual image titles."""
    img_w, img_h = (page_size[0] - 2 * margin, (page_size[1] - 4 * margin) // 2)
    page = Image.new('RGB', page_size, color='white')
    draw = ImageDraw.Draw(page)

    try:
        font_heading = ImageFont.truetype("DejaVuSans-Bold.ttf", 40)
        font_title = ImageFont.truetype("DejaVuSans.ttf", 18)
        font_subtext = ImageFont.truetype("DejaVuSans.ttf", 22)
    except:
        font_heading = font_title = ImageFont.load_default()

    y = margin

    # Folder-level heading (top of first page per group)
    if heading:
        draw.text((margin, y), heading, fill='black', font=font_heading)
        y += 50

    if subtext:
        draw.text((margin, y), subtext, fill='gray', font=font_subtext)
        y += 35
        
    # Draw each image with its title above it
    for i, img in enumerate(images):
        title = titles[i] if titles else f"Image {i+1}"

        # Draw title
        draw.text((margin, y), title, fill='black', font=font_title)
        y += 35  # space after title

        # Resize and paste image
        resized = img.resize((img_w, img_h))
        page.paste(resized, (margin, y))
        y += img_h + margin

    return page

def make_summary_pdf(base_path, output_pdf):
    from PIL import Image
    folder_metadata = {
        'QA_acquisition': {
            'heading': 'Acquisition',
            'subtext': 'Check noise floor is lower than signal, specially for high bvals. \n Check bvecs is sampled on the whole sphere.'
        },
        'QA_denoise': {
            'heading': 'Denoising',
            'subtext': 'Check residuals do not have structure, check sigma map is uniform'
        },
        'QA_gc': {
            'heading': 'Gibbs unringing',
            'subtext': 'Check that rings were removed'
        },
        'QA_topup': {
            'heading': 'Topup',
            'subtext': 'Check that the image is corrected for distortions'
        },
        'QA_eddy': {
            'heading': 'Eddy',
            'subtext': 'Check registration quality and motion correction'
        },
        'QA_dti_before_eddy': {
            'heading': 'DTI before eddy',
            'subtext': ''
        },
        'QA_dti_after_eddy': {
            'heading': 'DTI after eddy',
            'subtext': 'Tensor reconstruction after eddy correction; should match pre-eddy'
        },
        'QA_mask': {
            'heading': 'Masks',
            'subtext': 'Check that mask after preprocessing is good anf fits well the brain'
        }
    }



    pages = []

    for folder_name, meta in folder_metadata.items():
        folder_path = os.path.join(base_path, folder_name)
        png_paths = sorted(glob.glob(os.path.join(folder_path, '*.png')))
        if not png_paths:
            continue
    
        imgs = [Image.open(p).convert('RGB') for p in png_paths]
        titles = [os.path.basename(p) for p in png_paths]
    
        for i in range(0, len(imgs), 2):
            chunk_imgs = imgs[i:i+2]
            chunk_titles = titles[i:i+2]
            page = create_composite_page(
                chunk_imgs,
                titles=chunk_titles,
                heading=meta['heading'] if i == 0 else None,
                subtext=meta['subtext'] if i == 0 else None
            )
            pages.append(page)
    
        if not pages:
            raise ValueError("No PNGs found in QA folders.")

    first_page, rest = pages[0], pages[1:]
    first_page.save(output_pdf, save_all=True, append_images=rest)
    print(f"Saved PDF to: {output_pdf}")

    
##### TOPUP #####


def topup_routine(dwi_path, bids_strc, topupcfg_path, cfg):

    topup_input_files = create_topup_input_files(bids_strc, topupcfg_path, cfg)
    do_topup(topup_input_files, cfg)
    apply_topup(topup_input_files, dwi_path, bids_strc, cfg)


def create_topup_input_files(bids_strc, topupcfg_path,cfg):

    topup_input_files = {}

    # Forward and reverse b0 images
    concat_niftis([bids_strc.get_path('b0_fwd.nii.gz'), bids_strc.get_path('b0_rev.nii.gz')],
                  bids_strc.get_path('b0_fwd_rev.nii.gz'), 'all')
    
    topup_input_files['b0_fwd_rev'] = bids_strc.get_path('b0_fwd_rev.nii.gz')

    if any(dim <= 20 for dim in nib.load(bids_strc.get_path('b0_fwd_rev.nii.gz')).shape[:3]):
        print('Your data is a slab and that is not good for topup and eddy, we are padding it with zeros')
        data = bids_strc.get_path('b0_fwd_rev.nii.gz')
        data_pad = data.replace('.nii.gz','_padded.nii.gz')
        pad_image(data, data_pad)
        topup_input_files['b0_fwd_rev'] = data_pad

    # Acqp file
    nii_fwd = nib.load(bids_strc.get_path('b0_fwd.nii.gz')).get_fdata()
    nii_rev = nib.load(bids_strc.get_path('b0_rev.nii.gz')).get_fdata()
    
    # ensure there is a 4th dimention
    nii_fwd = np.expand_dims(nii_fwd, axis=-1) if nii_fwd.ndim == 3 else nii_fwd
    nii_rev = np.expand_dims(nii_rev, axis=-1) if nii_rev.ndim == 3 else nii_rev

    no_fwd = nii_fwd.shape[-1]
    no_rev = nii_rev.shape[-1]

    for ii in range(no_fwd):
        if ii == 0:
            write_txt([0, 1, 0, 0.01, '\n'], bids_strc.get_path('acqp_topup.txt'), 'w')  # 0.087

        else:
            write_txt([0, 1, 0, 0.01, '\n'], bids_strc.get_path('acqp_topup.txt'), 'a')

    for ii in range(no_rev):
        write_txt([0, -1, 0, 0.01, '\n'],
                  bids_strc.get_path('acqp_topup.txt'), 'a')

    topup_input_files['acqp'] = bids_strc.get_path('acqp_topup.txt')

    # Configuration file - rita
    copy_files([topupcfg_path], [bids_strc.get_path('mycnf_topup.cnf')])
    topup_input_files['config'] = bids_strc.get_path('mycnf_topup.cnf')
    topup_input_files['output_pref'] = bids_strc.get_path('b0_topup')

    return topup_input_files


def do_topup(topup_input_files, cfg):  # {config} rita

    fwd_rev_nii = topup_input_files['b0_fwd_rev']
    acqp = topup_input_files['acqp']
    config = topup_input_files['config']
    output_pref = topup_input_files['output_pref']

    exe =  os.path.join(cfg["fsl_path"], "topup")
    call = [exe,
            f'--imain={fwd_rev_nii}',
            f'--datain={acqp}',
            f'--config={config}',
            f'--out={output_pref}']

    print(' '.join(call))
    os.system(' '.join(call))
    

def apply_topup(topup_input_files, dwi_path, bids_strc, cfg):

    imain = dwi_path
    datain = topup_input_files['acqp']
    inindex = 1
    topup = bids_strc.get_path('b0_topup_fieldcoef')
    out = dwi_path.replace('.nii.gz', '_topup.nii.gz')

    if any(dim <= 20 for dim in nib.load(imain).shape[:3]):
         print('Your data is a slab and that is not good for topup and eddy, we are padding it with zeros')
         pad_image(imain, imain.replace('.nii.gz','_padded.nii.gz'))
         imain = imain.replace('.nii.gz','_padded.nii.gz')
     
    exe =  os.path.join(cfg["fsl_path"], "applytopup")
    call = [exe,
            f'--imain={imain}',
            f'--datain={datain}',
            f'--inindex={inindex}',
            f'--topup={topup}',
            f'--method=jac',
            f'--out={out}']

    print(' '.join(call))
    os.system(' '.join(call))
    
    if 'padded' in topup_input_files['b0_fwd_rev']:
        copy_file([out],[out.replace('.nii.gz','_padded.nii.gz')])
        unpad_image(out,out)


##### EDDY #####


def eddy_routine(dwi_path, out_path, mask_path, bvals_path, bvecs_path, topupon, cfg):

    eddy_input_files = create_eddy_input_files(
        dwi_path, out_path, mask_path, bvals_path, bvecs_path, topupon)
    do_eddy(eddy_input_files, cfg)


def create_eddy_input_files(dwi_path, out_path, mask_path, bvals_path, bvecs_path, topupon):

    eddy_input_files = {}

    basename = get_filename(bvals_path)
    basename = basename.replace('bvalsNom', '')

    nii_dwi = nib.load(dwi_path).get_fdata()
    no_dwi = nii_dwi.shape[-1]

    # Acqp file
    write_txt([0, 1, 0, 0.01], dwi_path.replace(get_filename(dwi_path) + '.nii.gz', basename +
              # last number is "dwell time" multiplied by "number of PE steps - 1"
                                                'acqp_eddy.txt'), 'w')
    eddy_input_files['acqp'] = dwi_path.replace(
        get_filename(dwi_path) + '.nii.gz', basename + 'acqp_eddy.txt')
    # Index file
    write_txt(['1'] * no_dwi, dwi_path.replace(get_filename(dwi_path) +
              '.nii.gz', basename + 'index_eddy.txt'), 'w')
    eddy_input_files['index'] = dwi_path.replace(
        get_filename(dwi_path) + '.nii.gz', basename + 'index_eddy.txt')
    # Bvals
    eddy_input_files['bvals'] = bvals_path
    # Bvecs
    eddy_input_files['bvecs'] = bvecs_path
    # dwi
    eddy_input_files['dwi'] = dwi_path

    # Mask
    if not os.path.exists(mask_path):
        mask_array = np.ones(nii_dwi.shape[0:3])
        array_to_nii(dwi_path, mask_array, dwi_path.replace(
            get_filename(dwi_path), basename + 'mask_before_ec'))
        eddy_input_files['mask'] = dwi_path.replace(
            get_filename(dwi_path), basename + 'mask_before_ec')
    else:
        eddy_input_files['mask'] = mask_path

    # adapted by rita to use eddy Quad as there is a .nii.gz uncessary
    eddy_input_files['out'] = out_path.replace('.nii.gz', '')

    # Check topup files exist
    if os.path.exists(dwi_path.replace(get_filename(dwi_path), basename + 'b0_topup_fieldcoef')) and topupon:
        eddy_input_files['topup'] = dwi_path.replace(
            get_filename(dwi_path) + '.nii.gz', basename + 'b0_topup')
    elif not os.path.exists(dwi_path.replace(get_filename(dwi_path), basename + 'b0_topup_fieldcoef')) and topupon:
        print(f"Error: The file at '{dwi_path.replace(basename, 'b0_topup_fieldcoef')}' does not exist.")
        sys.exit(1)

    return eddy_input_files


def do_eddy(eddy_input_files, cfg):  # rita addes repol and slm linear

    acqp = eddy_input_files['acqp']
    index = eddy_input_files['index']
    bvals = eddy_input_files['bvals']
    bvecs = eddy_input_files['bvecs']
    mask = eddy_input_files['mask']
    output = eddy_input_files['out']
    dwi = eddy_input_files['dwi']
        
    # If the image is like a slab (one dimension is very short), pad the image with zeros along 
    # that dimension otherwise eddy has problems and crashes.
    # I dont understand really deeply the cause of this problem but this seems
    # to be a good workaround
    if any(dim <= 20 for dim in nib.load(mask).shape[:3]):
        dwi_pad = dwi.replace('.nii.gz','_padded.nii.gz')
        mask_pad = mask.replace('.nii.gz','_padded.nii.gz')
        pad_image(mask, mask_pad)
        pad_image(dwi, dwi_pad)
        dwi = dwi_pad
        mask = mask_pad
        output_orig = output
        output = output+ '_padded'
        have_to_unpad = True
    else:
        have_to_unpad = False
      
    # Write call to eddy
    exe =  os.path.join(cfg["fsl_path"], "eddy_cuda10.2")
    call = [exe,
            f'--imain={dwi}',
            f'--mask={mask}',
            f'--index={index}',
            f'--acqp={acqp}',
            f'--bvecs={bvecs}',
            f'--bvals={bvals}', \
            f'--out={output}', \
            f'--repol', \
            #f'--ol_nstd=2',\
            f'--data_is_shelled --verbose']
        
    # If data are not acquired over the full sphere, use linear SLM
    if cfg['acq_wholesphere'] == 0:
        call.append('--slm=linear')

    # If there is topup data, use it
    if eddy_input_files.get('topup'):
        topup = eddy_input_files['topup']
        call.insert(6, f'--topup={topup}')

    # Run eddy
    print(' '.join(call))
    os.system(' '.join(call))
    
    # Unpad the images for the next steps
    if have_to_unpad:
        unpad_image(output + '.nii.gz', output_orig +'.nii.gz')
        copy_files([output + '.eddy_rotated_bvecs'], [output_orig + '.eddy_rotated_bvecs'])
        

def pad_image(img_path, out_path, pad_before=5, pad_after=5):
    img = nib.load(img_path)
    data = img.get_fdata()
    affine = img.affine
    header = img.header
 
    shape = data.shape
    spatial_shape = shape[:3]
 
    # Determine the smallest dimension index (0=X, 1=Y, 2=Z)
    smallest_axis = np.argmin(spatial_shape)
 
    # Construct padding for each axis
    if data.ndim == 3:
        padding = [(0, 0), (0, 0), (0, 0)]
    elif data.ndim == 4:
        padding = [(0, 0), (0, 0), (0, 0), (0, 0)]
    else:
        raise ValueError(f"Unsupported image dimension: {data.ndim}")
 
    # Apply padding only to the smallest axis
    padding[smallest_axis] = (pad_before, pad_after)
 
    padded_data = np.pad(data, tuple(padding), mode='constant', constant_values=0)
    padded_img = nib.Nifti1Image(padded_data, affine, header)
    nib.save(padded_img, out_path)
      
    
def unpad_image(img_path, out_path, pad_before=5, pad_after=5):
    img = nib.load(img_path)
    data = img.get_fdata()
    affine = img.affine
    header = img.header

    shape = data.shape
    spatial_shape = shape[:3]

    # Determine the smallest spatial axis (0=X, 1=Y, 2=Z)
    smallest_axis = np.argmin(spatial_shape)

    # Define slicing to remove padding
    slices = [slice(None)] * data.ndim  # e.g., [slice(None), slice(None), slice(None)] or 4D
    slices[smallest_axis] = slice(pad_before, spatial_shape[smallest_axis] - pad_after)

    unpadded_data = data[tuple(slices)]
    unpadded_img = nib.Nifti1Image(unpadded_data, affine, header)
    nib.save(unpadded_img, out_path)
    
##### QA #####


def QA_DTI_fit(nifti_path, bvals_path, bvecs_path, mask_path, output_path, cfg):

    create_directory(output_path)
    out_base = os.path.join(output_path, 'dti')

    exe =  os.path.join(cfg["fsl_path"], "dtifit")
    call = [exe,
            f'--data={nifti_path}',
            f'--mask={mask_path}',
            f'--bvecs={bvecs_path}',
            f'--bvals={bvals_path}',
            f'--out={out_base}',]
    print(' '.join(call))
    os.system(' '.join(call))
    
    FA = os.path.join(output_path,'dti_FA.nii.gz')
    V1 = os.path.join(output_path,'dti_V1.nii.gz')
    png_path = os.path.join(output_path,'V1.png')
    
    # Define threhsold intensity and voxels to plot
    dim1    = int(np.ceil(nib.load(FA).shape[0]/2))
    dim2    = int(np.ceil(nib.load(FA).shape[1]/2))
    dim3    = int(np.ceil(nib.load(FA).shape[2]/2))
    
    exe =  os.path.join(cfg["fsl_path"], "fsleyes")
    call = [f'{exe} render --hideCursor --hidex --voxelLoc {dim1} {dim2} {dim3} ',
            f'--xcentre -0 0 --ycentre -0 0 --zcentre -0 0 --labelSize 30',
            f'--outfile {png_path}',
            f'{FA}',
            f'{V1} --overlayType rgbvector '
            f'--modulateImage {FA}']
    
    print(' '.join(call))
    os.system(' '.join(call))

def QA_brain_extract(anat_path,output_path,anat_format,cfg):
    
    create_directory(output_path)
    
    slice_dim = cfg.get('slice_dim', 1)
    
    if cfg['slice_dim'] == 1:
        hide = '--hidex --hidez'
    elif cfg['slice_dim'] == 2:
        hide = '--hidex --hidey'
        
    img     = nib.load(anat_path)
    dim1    = int(np.ceil(img.shape[0]/2))
    dim2  = int(np.ceil(img.shape[1]/2))
    dim3    = int(np.ceil(img.shape[2]/2))
    maxint  = int(round(np.ceil(np.max(img.get_fdata())),1))

    
    png_path = os.path.join(output_path, f'{anat_format}.png')
    exe =  os.path.join(cfg["fsl_path"], "fsleyes")
    call = [f'{exe} render --hideCursor {hide} --voxelLoc {dim1} {dim2} {dim3} ',
            f'--xcentre -0 0 --ycentre -0 0 --zcentre -0 0 --labelSize 30 ',
            f'--outfile {png_path}',
            f'{anat_path}',
            f'-dr 0 {maxint} ',]

    print(' '.join(call))
    os.system(' '.join(call))
    
    
    anat_brain_path = anat_path.replace('.nii.gz','_brain.nii.gz')
    png_path = os.path.join(output_path, f'{anat_format}_brain.png')
    call = [f'{exe} render --hideCursor  {hide}  --voxelLoc {dim1} {dim2} {dim3} ',
            f'--xcentre -0 0 --ycentre -0 0 --zcentre -0 0 --labelSize 30 ',
            f'--outfile {png_path}',
            f'{anat_brain_path}',
            f'-dr 0 {maxint} ',]

    print(' '.join(call))
    os.system(' '.join(call))
    
    # make countour mask
    anat_brain_mask_path     = anat_path.replace('.nii.gz','_brain_mask.nii.gz')
    anat_brain_mask_dil_path = anat_brain_mask_path.replace('.nii.gz','_dil.nii.gz');
    dilate_im(anat_brain_mask_path, anat_brain_mask_dil_path, '1',cfg)
    countour_path = anat_brain_path.replace('.nii.gz', '_contour.nii.gz')
    exe2 = os.path.join(cfg["fsl_path"], "fslmaths")
    call = [exe2,
            f'{anat_brain_mask_dil_path}',
            f'-add',
            f'{anat_brain_mask_path}',
            f'-uthr 1',
            f'{countour_path}']
    print(' '.join(call))
    os.system(' '.join(call))


    png_path = os.path.join(output_path, f'{anat_format}_with_{anat_format}brain.png')
    call = [f'{exe} render --hideCursor  {hide}  --voxelLoc {dim1} {dim2} {dim3} ',
            f'--xcentre -0 0 --ycentre -0 0 --zcentre -0 0 --labelSize 30 ',
            f'--outfile {png_path}',
            f'{anat_path}',
            f'-dr 0 {maxint} ',
            f'{countour_path}']

    print(' '.join(call))
    os.system(' '.join(call))
  

def QA_mask(dwi, mask, mask_dil, mask_orig, output_path, cfg):
    
    create_directory(output_path)

    # Define threhsold intensity and voxels to plot
    dim1    = int(np.ceil(nib.load(mask).shape[0]/2))
    dim2    = int(np.ceil(nib.load(mask).shape[1]/2))
    dim3    = int(np.ceil(nib.load(mask).shape[2]/2))
    
    create_directory(output_path)
    
    png_path = os.path.join(output_path, 'dwi_mask_afterproc.png')
    exe =  os.path.join(cfg["fsl_path"], "fsleyes")
    call = [f'{exe} render --hideCursor --hidex --voxelLoc {dim1} {dim2} {dim3} ',
            f'--xcentre -0 0 --ycentre -0 0 --zcentre -0 0 --labelSize 30 ',
            f'--outfile {png_path}',
            f'{mask} ',
            f'-dr 0 1']

    print(' '.join(call))
    os.system(' '.join(call))
    
    png_path = os.path.join(output_path, 'dwi_mask_beforeproc.png')
    call = [f'{exe} render --hideCursor --hidex   --voxelLoc {dim1} {dim2} {dim3} ',
            f'--xcentre -0 0 --ycentre -0 0 --zcentre -0 0 --labelSize 30 ',
            f'--outfile {png_path}',
            f'{mask_orig} ',
            f'-dr 0 1']

    print(' '.join(call))
    os.system(' '.join(call))
    
  
    # make countour mask
    countour = os.path.basename(mask).replace('.nii.gz','_countour.nii.gz')
    exe2 = os.path.join(cfg["fsl_path"], "fslmaths")
    call = [exe2,
            f'{mask_dil}',
            f'-add',
            f'{mask}',
            f'-uthr 1',
            f'{countour}']
    print(' '.join(call))
    os.system(' '.join(call))


    png_path = os.path.join(output_path, 'dwi_with_mask_afterproc.png')
    maxint  = int(round(np.ceil(np.max(nib.load(dwi).get_fdata())),1))
    call = [f'{exe} render --hideCursor --hidex  --voxelLoc {dim1} {dim2} {dim3} ',
            f'--xcentre -0 0 --ycentre -0 0 --zcentre -0 0 --labelSize 30 ',
            f'--outfile {png_path}',
            f'{dwi}',
            f'-dr 0 {maxint} ',
            f'{countour}']

    print(' '.join(call))
    os.system(' '.join(call))
    
def QA_denoise(bids_strc, res, sigma, output_path, cfg):
    
    res_path    = bids_strc.get_path(res)
    sigma_path  = bids_strc.get_path(sigma)
    
    slice_dim = cfg.get('slice_dim', 1)
    if cfg['slice_dim'] == 1:
        hide = '--hidex --hidez'
    elif cfg['slice_dim'] == 2:
        hide = '--hidex --hidey'
        
    # Define threhsold intensity and voxels to plot
    maxintsigma = int(round(0.9 * np.max(np.max(nib.load(sigma_path).get_fdata()))))
    maxintres   = int(round(0.9 * np.max(np.max(nib.load(res_path).get_fdata()))))
    dim1    = int(np.ceil(nib.load(res_path).shape[0]/2))
    dim2    = int(np.ceil(nib.load(res_path).shape[1]/2))
    dim3    = int(np.ceil(nib.load(res_path).shape[2]/2))
    
    create_directory(output_path)
    
    png_path = os.path.join(output_path, res.replace('.nii.gz','.png'))
    exe =  os.path.join(cfg["fsl_path"], "fsleyes")
    call = [f'{exe} render --hideCursor {hide} --voxelLoc {dim1} {dim2} {dim3} ',
            f'--xcentre -0 0 --ycentre -0 0 --zcentre -0 0 --labelSize 30 ',
            f'--outfile {png_path}',
            f'{res_path} ',
            f'-dr -{maxintres} {maxintres}']

    print(' '.join(call))
    os.system(' '.join(call))
    
    png_path = os.path.join(output_path, sigma.replace('.nii.gz','.png'))
    call = [f'{exe} render --hideCursor {hide} --voxelLoc {dim1} {dim2} {dim3} ',
            f'--xcentre -0 0 --ycentre -0 0 --zcentre -0 0 --labelSize 30 ',
            f'--outfile {png_path}',
            f'{sigma_path} --cmap red-yellow',
            f'-dr 0 {maxintsigma}']

    print(' '.join(call))
    os.system(' '.join(call))
    

def QA_topup(bids_strc, before, after, output_path, cfg):

    
    before_path = bids_strc.get_path(before)
    after_path = bids_strc.get_path(after)
    
    # Define threhsold intensity and voxels to plot
    maxint  = int(round(0.8 * np.max(np.max(nib.load(before_path).get_fdata()))))
    dim1    = int(np.ceil(nib.load(before_path).shape[0]/2))
    dim2    = int(np.ceil(nib.load(before_path).shape[1]/2))
    dim3    = int(np.ceil(nib.load(before_path).shape[2]/2))
    
    create_directory(output_path)
    
    png_path = os.path.join(output_path, before.replace('.nii.gz','.png'))
    exe =  os.path.join(cfg["fsl_path"], "fsleyes")
    call = [f'{exe} render --hideCursor --hidex  --voxelLoc {dim1} {dim2} {dim3}',
            f'--xcentre -0 0 --ycentre -0 0 --zcentre -0 0 --labelSize 30 ',
            f'--outfile {png_path}',
            f'{before_path} ',
            f'-dr 0 {maxint} ']

    print(' '.join(call))
    os.system(' '.join(call))
    
    png_path = os.path.join(output_path, after.replace('.nii.gz','.png'))
    call = [f'{exe} render --hideCursor --hidex  --voxelLoc {dim1} {dim2} {dim3}',
            f'--xcentre -0 0 --ycentre -0 0 --zcentre -0 0 --labelSize 30 ',
            f'--outfile {png_path}',
            f'{after_path} ',
            f'-dr  0 {maxint} ']

    print(' '.join(call))
    os.system(' '.join(call))
    
    
def QA_gc(bids_strc, before, after, output_path, cfg):
    
    before_path = bids_strc.get_path(before)
    after_path = bids_strc.get_path(after)
    
    maxint = int(round(0.8* np.max(np.max(nib.load(before_path).get_fdata()))))
    dim1    = int(np.ceil(nib.load(before_path).shape[0]/2))
    dim2    = int(np.ceil(nib.load(before_path).shape[1]/2))
    dim3    = int(np.ceil(nib.load(before_path).shape[2]/2))
    
    create_directory(output_path)
    
    png_path = os.path.join(output_path, before.replace('.nii.gz','.png'))
    exe =  os.path.join(cfg["fsl_path"], "fsleyes")
    call = [f'{exe} render --hideCursor --hidex   --voxelLoc {dim1} {dim2} {dim3} ',
            f'--xcentre -0 0 --ycentre -0 0 --zcentre -0 0 --labelSize 30 ',
            f'--outfile {png_path}',
            f'{before_path}',
            f'-dr 0 {maxint} ']

    print(' '.join(call))
    os.system(' '.join(call))
    
    png_path = os.path.join(output_path, after.replace('.nii.gz','.png'))
    call = [f'{exe} render --hideCursor --hidex --voxelLoc  {dim1} {dim2} {dim3}',
            f'--xcentre -0 0 --ycentre -0 0 --zcentre -0 0 --labelSize 30 ',
            f'--outfile {png_path}',
            f'{after_path} ',
            f'-dr 0 {maxint} ']

    print(' '.join(call))
    os.system(' '.join(call))
     
   

def QA_eddy(mask_path, mask_dil_path, dwi_path, dwi_ec_path, output_path, bvals_path, bids_strc, cfg):

    create_directory(output_path)

    # make countour mask
    countour_path = mask_path.replace('.nii.gz', '_bo_contour.nii.gz')
    exe =  os.path.join(cfg["fsl_path"], "fslmaths")
    call = [exe,
            f'{mask_dil_path}',
            f'-add',
            f'{mask_path}',
            f'-uthr 1',
            f'{countour_path}']
    print(' '.join(call))
    os.system(' '.join(call))

    # retreive bvals = b0 position and chose those volumes
    bvals = np.loadtxt(bvals_path)
    b0s = np.where(bvals == 0)
    maxint = int(round(0.8* np.max(np.max(nib.load(dwi_path).get_fdata()))))
    dim1    = int(np.ceil(nib.load(mask_path).shape[0]/2))
    dim2    = int(np.ceil(nib.load(mask_path).shape[1]/2))
    dim3    = int(np.ceil(nib.load(mask_path).shape[2]/2))

    for ii in np.linspace(1, len(b0s[0])-1, num=3, dtype='int'):
        volume = b0s[0][ii]

        # plot dwi before eddy
        png_path = os.path.join(
            output_path, 'nodifcontour_v' + str(volume) + '_dwi.png')
        exe =  os.path.join(cfg["fsl_path"], "fsleyes")
        call = [f'{exe} render --hideCursor --hidex  --voxelLoc {dim1} {dim2} {dim3} ',
                f'--xcentre -0 0 --ycentre -0 0 --zcentre -0 0 --labelSize 30 ',
                f'--outfile {png_path}',
                f'{dwi_path}',
                f'-v {volume}',
                f'-dr 0 {maxint} ',
                f'{countour_path}']

        print(' '.join(call))
        os.system(' '.join(call))

        # plot dwi after eddy
        png_path = os.path.join(
            output_path, 'nodifcontour_v' + str(volume) + '_eddy.png')
        call = [f'{exe} render --hideCursor --hidex --voxelLoc {dim1} {dim2} {dim3}',
                f'--xcentre -0 0 --ycentre -0 0 --zcentre -0 0 --labelSize 30 ',
                f'--outfile {png_path}',
                f'{dwi_ec_path}',
                f'-v {volume}',
                f'-dr 0 {maxint}',
                f'{countour_path}']
        print(' '.join(call))
        os.system(' '.join(call))
          
    output_qc = os.path.join(output_path,'qc')
    if os.path.exists(output_qc):
        shutil.rmtree(output_qc)
    
    indx_eddy = bids_strc.get_path('index_eddy.txt')
    acqc_eddy = bids_strc.get_path('acqp_eddy.txt')
    suffix    = dwi_ec_path.replace('.nii.gz', '')

    exe =  os.path.join(cfg["fsl_path"], "eddy_quad")
    call = [f'{exe} {suffix}', \
            f'-idx {indx_eddy} -par {acqc_eddy}', \
            f'-m {mask_path} -b {bvals_path}', \
            f'-o {output_qc}']
    print(' '.join(call))
    os.system(' '.join(call))
        

def QA_reg(moving_path, fixed_path, output_path, cfg):

    create_directory(output_path)
    os.chdir(output_path)
    exe =  os.path.join(cfg["fsl_path"], "slicer")
    call = [exe,
            f'{fixed_path}',
            f'{moving_path}',
            f'-s 2 -y 0.35 slb.png -y 0.45 slc.png -y 0.55 sld.png -y 0.65 sle.png']

    os.system(' '.join(call))


    out_image = os.path.join(output_path,'Mov2Fixed.png')
    exe =  os.path.join(cfg["fsl_path"], "pngappend")
    call = [exe,
            'slb.png + slc.png + sld.png + sle.png',
            f'{out_image}; rm sl*.png']
    os.system(' '.join(call))


def plot_bvals(bids_strc):

    bvals_nom = read_numeric_txt(bids_strc.get_path('bvalsNom.txt'))
    bvals_eff = read_numeric_txt(bids_strc.get_path('bvalsEff.txt'))
    bvals_avg = calc_avg_bval(bvals_nom, bvals_eff)

    bvals_nom_unique = np.unique(bvals_nom)

    # QA plot: effective bvals as a function of nominal bvals
    fig, ax = plt.subplots()
    # Plot the data and save figure
    ax.plot(bvals_nom, bvals_eff, 'bo', markersize=8)
    # Plot only the mean values
    ax.plot(bvals_nom_unique, bvals_avg, 'ro', markersize=8)
    # Annotate the mean values
    for ii in range(len(bvals_avg)):
        ax.annotate(str(round(bvals_avg[ii], 1)), xy=(bvals_nom_unique[ii], bvals_avg[ii]),
                    xytext=(bvals_nom_unique[ii] - 100.0, bvals_avg[ii] + 50.0))
    # Set axes
    ax.set_xlabel('Nominal b-val',
                  fontdict={'size': 12, 'weight': 'bold', 'style': 'italic'})
    ax.set_ylabel('Effective b-val',
                  fontdict={'size': 12, 'weight': 'bold', 'style': 'italic'})
    ax.grid(True)
    plt.savefig(bids_strc.get_path('bvals_Eff_Nom.png'))
                #bbox_inches='tight', dpi=300)
    plt.close()


def QA_plotbvecs(bvec_path, bval_path, output_path):
    import distinctipy

    create_directory(output_path)

    # Load the bvecs file (assuming it is a whitespace-delimited text file)
    bvecs = np.loadtxt(bvec_path)
    bvals = np.loadtxt(bval_path)
    unique_bvals = np.unique(bvals)
    
    ndir_per_shell = []
    for i in range(len(unique_bvals)):
        a=list(bvals)
        ndir_per_shell.append(a.count(unique_bvals[i]))
    
    # normalize to the outter shell
    bvals_norm = np.multiply(bvals,1/max(bvals))
    bvecs_norm = np.multiply(bvecs,np.sqrt(bvals_norm))
    
    # create colors
    #color_list = distinctipy.get_colors(len(unique_bvals),pastel_factor=0.5)
    color_list = [(0.3760998343354473, 0.35126571526332956, 0.8197999869450858),
     (0.42466800911865693, 0.9945497644533038, 0.3769543628852434),
     (0.9828680799627539, 0.523830954896532, 0.3985256551508593),
     (0.40949821871153147, 0.8205446136451112, 0.9519624424548683),
     (0.4043490130959444, 0.5346375952340997, 0.3376838651160777),
     (0.8706067111878107, 0.5495953100182577, 0.995103250032075)]

    # Create a 3D plot
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')
    handles = []
    
    # Plot the data
    for k, b in enumerate(unique_bvals):
        bvecs_to_plot = bvecs_norm[:,bvals==b]
        sc = ax.scatter(
            bvecs_to_plot[0, :],
            bvecs_to_plot[1, :],
            bvecs_to_plot[2, :],
            color=color_list[k],
            edgecolor=color_list[k],
            alpha=1,
            marker='o',
            s=8,
            label=f"b = {b} ms/µm²"
        )
        handles.append(sc)
       # ax.scatter(bvecs_to_plot[0, :], bvecs_to_plot[1, :], bvecs_to_plot[2, :], color=color_list[k], marker='*')

        
    # Add legend
    ax.legend(handles=handles, loc='center left', bbox_to_anchor=(0.8, 0.5), fontsize=8,
              handletextpad=0.05,   # space between marker and text (horizontal)
              labelspacing=0.1,
              borderpad=0.1,
              borderaxespad=0.3)

    # Set axis limits
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.grid(True)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    #ax.set_axis_off()

    # Set axis properties for better visualization
    ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio
    plt.savefig(os.path.join(output_path, 'QA_bvecs.png'),
                bbox_inches='tight', dpi=300)


def QA_plotSNR(bids_strc, dwi, snr, dwi_sigma, mask, bvals, output_path):

    # Load data
    bvals_nom  = read_numeric_txt(bids_strc.get_path(bvals))
    mask       = nib.load(bids_strc.get_path(mask)).get_fdata()

    create_directory(output_path)

    # Compute noise floor
    nf        = nib.load(bids_strc.get_path(dwi_sigma)).get_fdata()*np.sqrt(np.pi/2)
    nf_masked = nf * mask
    nf_masked = nf.reshape(nf.shape[0]*nf.shape[1]*nf.shape[2], 1)
    nf_masked = nf_masked[~(np.isnan(nf_masked).any(axis=1) | (nf_masked == 0).any(axis=1))]

    # Plot SNR, masked by whole brain mask
    SNR = nib.load(bids_strc.get_path(snr)).get_fdata()
    for v in range(SNR.shape[-1]):
        SNR[:, :, :, v] = np.multiply(SNR[:, :, :, v], mask)

    data = SNR.reshape(SNR.shape[0]*SNR.shape[1]*SNR.shape[2], SNR.shape[3]);
    data[data == 0] = np.nan

    fig, ax = plt.subplots()
    ax.plot(np.transpose(bvals_nom), np.nanmean(data, axis=0), 'bo', markersize=3)
    ax.set_xlabel('Nominal b-val',
                  fontdict={'size': 12, 'weight': 'bold', 'style': 'italic'})
    ax.set_ylabel('Mean SNR', fontdict={
                  'size': 12, 'weight': 'bold', 'style': 'italic'})
    ax.grid(True)
    plt.savefig(os.path.join(output_path, 'QA_SNR.png'),
                bbox_inches='tight', dpi=300)
    
    
    # Plot S/S0, masked by whole brain mask
    dwi = nib.load(bids_strc.get_path(dwi)).get_fdata()
    
    for v in range(dwi.shape[-1]):
        dwi[:, :, :, v] = np.multiply(dwi[:, :, :, v], mask)

    data = dwi.reshape(dwi.shape[0]*dwi.shape[1]*dwi.shape[2], dwi.shape[3]);
    data[data == 0] = np.nan

    fig, ax = plt.subplots()
    ax.plot(np.transpose(bvals_nom), np.nanmean(data, axis=0), 'bo', markersize=3)
    ax.plot(np.transpose(bvals_nom), np.repeat(np.nanmean(nf_masked), np.transpose(bvals_nom).shape[0]))
    ax.set_xlabel('Nominal b-val',
                  fontdict={'size': 12, 'weight': 'bold', 'style': 'italic'})
    ax.set_ylabel('Mean S and noise floor', fontdict={
                  'size': 12, 'weight': 'bold', 'style': 'italic'})
    ax.grid(True)
    plt.savefig(os.path.join(output_path, 'QA_S_nf.png'),
                bbox_inches='tight', dpi=300)
    
def QA_ROIs(roi_paths, template, output_path, cfg):
   
    out_image = os.path.join(output_path, 'ROIs.png')
    out_image_labeled = os.path.join(output_path, 'ROIs_labeled.png')
    output_txt_path = os.path.join(output_path, 'mask_ROIs.txt')
    output_mask_path = os.path.join(output_path, 'mask_ROIs.nii.gz')
    
    # -------------- Merge ROIs into one label mask  --------------

    # Make mask all ROIs
    ref_img = nib.load(roi_paths[0])
    data_shape = ref_img.shape
    affine = ref_img.affine
    header = ref_img.header
    merged_mask = np.zeros(data_shape, dtype=np.uint8)
    label = 1
    label_mapping = []
    for roi_path in roi_paths:
        if 'WB' in roi_path:
            continue  # skip unwanted masks
    
        roi_img = nib.load(roi_path)
        roi_data = roi_img.get_fdata()
    
        # Binarize the ROI and label it (overwrite label only where it's 0)
        merged_mask[(roi_data > 0) & (merged_mask == 0)] = label
        roi_name = os.path.splitext(os.path.basename(roi_path))[0]
        label_mapping.append((label, roi_name))
        label += 1
        
    with open(output_txt_path, 'w') as f:
        for lab, roi_name in label_mapping:
            f.write(f"{lab}: {roi_name}\n")
   
    # Save merged mask
    merged_mask_img = nib.Nifti1Image(merged_mask, affine, header)
    nib.save(merged_mask_img, output_mask_path)
    n_labels = len(label_mapping)

    # -------------- Render with FSLeyes --------------
    
    # Make fsl eyes call
    exe =  os.path.join(cfg["fsl_path"], "fsleyes")
    call = [
        f'{exe}', 'render',
        '--scene', 'lightbox',
        '--outfile', out_image ,
        '--hideCursor',
        '--hidex',
        '--hidez',
        '--xcentre', '0', '0',
        '--ycentre', '0', '0',
        '--zcentre', '0', '0',
        '--labelSize', '30',
        '--sliceLocation',
        '--zrange','0.2','0.8',
        '--sliceSpacing','0.01',
        '--sampleSlices','centre',
        template,                
        '--name', 'Base',
    ]
    call.extend([
                os.path.join(output_path, 'mask_ROIs.nii.gz'),
                '--name', 'ROI',
                '--overlayType', 'volume',
                '--cmap', 'HSV',
                '--displayRange', '0',  str(n_labels),
                '--volume', '0',
                '--alpha', '70.0'
            ])

    
    print(' '.join(call))
    subprocess.run(call)
    
    # -------------- Create legend panel --------------
    from matplotlib import cm
    from PIL import Image, ImageDraw, ImageFont
    cmap = cm.get_cmap('hsv', n_labels + 1)

    base_img = Image.open(out_image).convert("RGB")
    w, h = base_img.size
    
    legend_width = 500
    margin = 30
    line_height = 40
    box_size = 24
    
    canvas = Image.new("RGB", (w + legend_width, h), "white")
    canvas.paste(base_img, (0, 0))
    
    draw = ImageDraw.Draw(canvas)
    
    # Try a nicer font, fallback if unavailable
    try:
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
        font_text = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    except:
        font_title = ImageFont.load_default()
        font_text = ImageFont.load_default()
    
    x0 = w + margin
    y = margin
    
    draw.text((x0, y), "ROI labels", fill="black", font=font_title)
    y += 50
    
    for lab, roi_name in label_mapping:
        rgba = cmap(lab)
        color = tuple(int(255 * c) for c in rgba[:3])
    
        # colored square
        draw.rectangle(
            [x0, y, x0 + box_size, y + box_size],
            fill=color,
            outline="black"
        )
    
        # label text
        roi_name = roi_name.replace(".nii.gz", "").replace(".nii", "")
        roi_name_clean = roi_name.split("_")[-1]
        draw.text(
            (x0 + box_size + 15, y - 2),
            f"{roi_name_clean}",
            fill="black",
            font=font_text
        )
    
        y += line_height
    
    canvas.save(out_image_labeled)
    print(f"Saved labeled image to: {out_image_labeled}")
    
def plot_summary_params_model(output_path, model, cfg, template_path=None, countour_path=None):
   from .custom_functions_models import get_param_names_model

   # Make colorbar
   import matplotlib.colors as mcolors
   import matplotlib.cm as cm
   import imutils
   from scipy.ndimage import rotate
   

   jet = cm.get_cmap('jet', 256)
   if 'Nexi' in model or 'Smex' in model:
       jet_colors = jet(np.linspace(0, 1, 256))
       fade_len = 20
       fade = np.linspace(0, 1, fade_len).reshape(-1, 1)
       jet_colors[:fade_len, :3] *= fade  # Keep alpha (4th channel) unchanged
       jet_colors[-fade_len:, :3] *= fade[::-1]  # Reverse fade for the end
       custom_jet_black = mcolors.ListedColormap(jet_colors)
   else:
       jet_colors = jet(np.linspace(0, 1, 256))
       fade_len = 1
       fade = np.linspace(0, 1, fade_len).reshape(-1, 1)
       jet_colors[:fade_len, :3] *= fade  # Keep alpha (4th channel) unchanged
       custom_jet_black = mcolors.ListedColormap(jet_colors)
   
   # Get model parameter names and display ranges
   patterns, lims, maximums = get_param_names_model(model,cfg['is_alive'])
   
   # Load contour data
   if countour_path is not None:
       countour_data = nib.load(countour_path).get_fdata()
       if cfg['slice_dim'] == 1 :
            slicee = int(np.ceil(nib.load(countour_path).shape[1]/2))
            #contour = imutils.rotate(countour_data[:,slicee, :], angle=90)
            contour = np.rot90(countour_data[:,:,slicee], k=1)
            fact = int((max(contour.shape) - min(contour.shape)) / 2)
            if  fact != 0:
                contour = contour[fact:-fact, :]
       elif cfg['slice_dim'] == 2:
            slicee = int(np.ceil(nib.load(countour_path).shape[2]/2))
            #contour = imutils.rotate(countour_data[:,:,slicee], angle=90)
            contour = np.rot90(countour_data[:,:,slicee], k=1)
            
       contour[np.isnan(contour)] = 0

   # Create subplot grid
   n_params = len(patterns)
   n_rows = 1 if n_params <= 4 else 2
   n_cols = math.ceil(n_params / n_rows)

   if template_path is not None:
       n_cols = math.ceil((n_params +1 )/ n_rows)
           
   fig, axs = plt.subplots(n_rows, n_cols, figsize=(12, 3.6))
   axs = axs.flatten()
   fig.subplots_adjust(wspace=0.05, hspace=0.11, top=0.92, bottom=0.1, left=0.05, right=0.95)
   
   # Display each parameter slice
   for ax, pattern, lim in zip(axs, patterns, lims):
       
       # Load file
       files = glob.glob(os.path.join(output_path, pattern))
       # Remove files with "initialization"
       matched_file = [f for f in files if "initialization" not in os.path.basename(f)]

      # matched_file = glob.glob(os.path.join(output_path, pattern))
       if pattern=='*sandi*f*':
           matched_file = [
                f for f in matched_file
                if 'fs' not in os.path.basename(f).lower()] 
       param_path = matched_file[0]
       param_data = nib.load(param_path).get_fdata()
       print(matched_file)
   
       # Extract and process middle slice
       if cfg['slice_dim'] == 1:
           slicee = int(np.ceil(nib.load(param_path).shape[1]/2))
           img = param_data[:,slicee, :]
           img[np.isnan(img)] = 0
           img = np.rot90(img, k=1)
           # fact = int((max(img.shape) - min(img.shape)) / 2)
           # if  fact != 0:
           #     img = img[fact:-fact, :]
    
       elif cfg['slice_dim'] == 2:
           slicee = int(np.ceil(nib.load(param_path).shape[2]/2))
           img = param_data[:,:,slicee]
           img[np.isnan(img)] = 0
           img = np.rot90(img, k=-1)
           #img = np.rot90(img, k=-1)

       # Show slice
       #if model in ['Nexi', 'Smex']:
         #  pattern = original_pattern
       im = ax.imshow(img, cmap=custom_jet_black, vmin=lim[0], vmax=lim[1])
       cleaned_pattern = re.sub(model, '', pattern, flags=re.IGNORECASE)
       cleaned_pattern = cleaned_pattern.replace('[^s]', '')
       cleaned_pattern = re.sub(r'\*{2,}', '*', cleaned_pattern).strip('*')
       ax.set_title(cleaned_pattern)
       ax.axis('off')
       
       # overlay countour
       if countour_path is not None:
           ax.contour(contour, levels=[0.5], colors='white', linewidths=1)
           #overlay = np.ma.masked_where(contour == 0, contour)
           #ax.imshow(overlay, cmap='gray', alpha=0.4)
   
       # Add colorbar
       cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.02)
       cbar.set_ticks([lim[0], lim[1]])
       cbar.ax.set_xticklabels([lim[0], lim[1]], rotation=-45)
       cbar.ax.tick_params(labelsize=10)
      
     
   # Add template
   if template_path is not None:
       template_data = nib.load(template_path).get_fdata()
       if cfg['slice_dim'] == 1 :
           slicee = int(np.ceil(nib.load(template_path).shape[1]/2))
           img = template_data[:,slicee, :]
           img[np.isnan(img)] = 0
           img = np.rot90(img, k=1)
           fact = int((max(img.shape) - min(img.shape)) / 2)
           # if  fact != 0:
           #     img = img[fact:-fact, :]
           maxint = int(np.round(0.9*np.ceil(np.max(img))))
       elif cfg['slice_dim'] == 2:
           slicee = int(np.ceil(nib.load(template_path).shape[2]/2))
           img = template_data[:,:,slicee]
           img[np.isnan(img)] = 0
           img = np.rot90(img, k=-1)
           #img = np.rot90(img, k=-1)
           maxint = int(np.round(0.9*np.ceil(np.max(img))))

       # Show slice
       ax = axs[-1]  # next axis after params
       im = ax.imshow(img, cmap='gray')
       ax.set_title('template')
       ax.axis('off')
       
       # overlay countour
       if countour_path is not None:
           ax.contour(contour, levels=[0.5], colors='white', linewidths=1)
    
       # Add colorbar
       cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.02)
       cbar.set_ticks([0, maxint])
       cbar.ax.set_xticklabels([], rotation=-45)
       cbar.ax.tick_params(labelsize=10)
       
   # Hide unused axes
   n_used = n_params + (1 if template_path is not None else 0)
   if len(axs) > n_used:
        for ax in axs[n_used:]:
            ax.set_visible(False)

   
   # Save and close figure
  # plt.tight_layout(rect=[0, 0, 1, 1])
   plt.savefig(os.path.join(output_path, f'{model}_summary.png'))
   plt.close(fig)

def plot_with_shade(ax, x, y, y_std, color, **kwargs):
    line, = ax.plot(x, y, c=color, **kwargs)
    # mask NaNs so fill_between doesn't choke
    #m = ~(np.isnan(y) | np.isnan(y_std))
    ax.fill_between(x, (y-y_std), (y+y_std),
                    color=color, alpha=0.20, linewidth=0, zorder=line.get_zorder()-1)
    return line
    
##### BVALS #####


def modify_units_bvals(path_bvals, new_path_bvals):

    with open(new_path_bvals, 'w') as f:
        f.write(' '.join(map(str, np.loadtxt(path_bvals)*1e-3)))
    f.close()


def param_extract(param_path, bvals_path, desiredbvals, output_path):

    param = np.loadtxt(param_path)
    bvals = np.loadtxt(bvals_path)
    new_param = param[np.in1d(bvals, desiredbvals)]

    with open(output_path, 'w') as f:
        f.write(' '.join(map(str, new_param)))
    f.close()


def prep_bval_file(input_path, output_path):

    #  Step 1: Load the file with numbers (assuming each number is space-separated)
    with open(input_path, 'r') as file:
        # Read the entire line and split it into a list of strings
        numbers = file.readline().split()

    # Step 2: Process the numbers (divide by 1000 and round to 1 decimal)
    processed_numbers = [round(float(num) / 1000, 1) for num in numbers]

    # Step 3: Save the processed numbers back into a new text file
    with open(output_path, 'w') as file:
        # Join the numbers back into a string with space separation and write to the file
        file.write(' '.join(map(str, processed_numbers)))


def calc_avg_bval(bvals_nom, bvals_eff):

    # Calculate average effective values for each nominal shell value
    unique_numbers = np.unique(bvals_nom)
    means = []
    for unique_number in unique_numbers:
        corresponding_indices = np.where(bvals_nom == unique_number)
        corresponding_values = bvals_eff[corresponding_indices]
        mean_value = np.mean(corresponding_values)
        means.append(mean_value)

    return means

##### STUDY OPERATIONS #####


def concat_param(excel_param, path_bvals, output_path):

    array_param = []
    for k in range(0, len(excel_param)):
        array_param.extend(
            np.repeat(excel_param[k], len(np.loadtxt(path_bvals[k]))))

        with open(output_path, 'w') as f:
            f.write(' '.join(map(str, array_param)))
        f.close()


def extract_scan_no(scan_list, scan_idx, study_scanMode, paths_fwd, paths_rev):

    if scan_list['acqType'][scan_idx] == study_scanMode:
        # Create list with the scan numbers
        if scan_list['phaseDir'][scan_idx] == 'fwd':
            if scan_list['preprocOrder'][scan_idx] == 'first':
                paths_fwd.insert(0, scan_list['scanNo'][scan_idx])
            else:
                paths_fwd.append(scan_list['scanNo'][scan_idx])
        if scan_list['phaseDir'][scan_idx] == 'rev':
            paths_rev.append(scan_list['scanNo'][scan_idx])
    return paths_fwd, paths_rev


def extract_methods(methods_in, bids_strc, acqp, cfg=None):

    if acqp == 'PGSE':

        bmatrix = []
        bvals_eff = []
        bvals_nom = []
        dirs = []
        no_dirs_pershell = []
        with open(methods_in, 'r') as f:
            for line in f:
                if '##TITLE=' in line:
                    PV_version = (line.split(' ')).pop().strip()
                if '##$PVM_NRepetitions=' in line:
                    no_rep = int(line.split('=')[1])
                if '##$PVM_DwNDiffExpEach=' in line:
                    noShells = int(line.split('=')[1])
                if 'PVM_DwNDiffExp=' in line:
                    no_dwi = int(line.split('=')[1])
                if '##$PVM_DwDir=' in line:
                    values = re.findall(r'\d+', line.split('=')[1])
                    dims_dirs = [int(x) for x in values]
                    line = next(f)
                    while not '##$PVM_DwDgSwitch=' in line:
                        dirs.extend([float(x) for x in line.split()])
                        line = next(f)
                if '##$PVM_DwBMat=' in line:
                    values = re.findall(r'\d+', line.split('=')[1])
                    dims_bmat = [int(x) for x in values]
                    line = next(f)
                    while not '##$PVM_DwBMatImag' in line:
                        bmatrix.extend([float(x) for x in line.split()])
                        line = next(f)
                if '##$PVM_DwAoImages=' in line:
                    no_b0 =  int(line.split('=')[1])
                if '##$PVM_DwShellNDir=' in line: 
                    next_line = next(f).strip()
                    no_dirs_pershell = [int(x) for x in next_line.split()]
                if 'PVM_DwBvalEach=' in line:
                    line = next(f)
                    while not '##$PVM_' in line:
                        bvals_nom.extend([float(x) for x in line.split()])
                        line = next(f)
                if '##$PVM_DwEffBval=' in line:
                    line = next(f)
                    while '##$PVM_' not in line:
                        for tok in line.split():
                            if tok.startswith('@'):
                                m = re.match(r'@(\d+)\*\(([^)]+)\)', tok)
                                n = int(m.group(1))
                                v = float(m.group(2))
                                bvals_eff.extend([v] * n)
                            else:
                                bvals_eff.append(float(tok))
                        line = next(f)
                        
        if PV_version == 'V1.1':

            # Reshape and duplicate initial dirs matrix for all the shells
            dirs = np.array(dirs).reshape(
                dims_dirs[1], dims_dirs[0], order='F')
            dirs = np.repeat(dirs, noShells, axis=1)
            no_dirs = dims_dirs[0]
            no_b0 = dims_bmat[0] - dirs.shape[1]

            # Pad with zeros for b=0 images
            dirs = np.concatenate((np.zeros((3, no_b0)), dirs), axis=1)
            # Repeat directions for each Nrep
            dirs = np.tile(dirs, no_rep)

            # Create nominal bvals
            bvals_nom = [x for i in range(no_dirs) for x in bvals_nom]
            bvals_nom = np.concatenate(
                (np.zeros((no_b0, 1)), np.array(bvals_nom).reshape(-1, 1))).T
            bvals_nom = np.tile(bvals_nom, no_rep)

            # Format effective bvals
            bvals_eff = np.tile(np.array(bvals_eff)[:, np.newaxis].T, no_rep)

            # Format bmatrix
            bmatrix = np.array(bmatrix).reshape(
                dims_bmat[1]*dims_bmat[2], dims_bmat[0], order='F')
            bmatrix = np.tile(bmatrix, no_rep)

            # Save all files
            np.savetxt(bids_strc.get_path('bvalsEff.txt'), bvals_eff, delimiter=' ', fmt='%.1f')
            np.savetxt(bids_strc.get_path('bvalsNom.txt'), bvals_nom, delimiter=' ', fmt='%.1f')
            np.savetxt(bids_strc.get_path('bvecs.txt'), dirs, delimiter=' ', fmt='%.16f')
            np.savetxt(bids_strc.get_path('bmatrix.txt'), bmatrix, delimiter=' ', fmt='%.16f')

        elif PV_version == 'V3.5' or PV_version == 'V3.6' or PV_version == 'V3.7': 
            # bvals,and direction are saved differently from previous version 
            # direction number is already the total number of directions in a multi-shell protocol, so no need to replicate for all shells

            # Reshape for 3 x number of dirs
            dirs                = np.array(dirs).reshape(dims_dirs[1], dims_dirs[0], order='F')
            no_dirs             = dims_dirs[0]
            
            # If only one value is provided, reshape and duplicate initial dirs matrix for all the shells
            if len(no_dirs_pershell) == 1:
                dirs = np.tile(dirs, (1, noShells))

            # Pad with zeros for b=0 images
            dirs                = np.concatenate((np.zeros((3, no_b0)), dirs), axis=1)
            
            # If only one value is provided, use it for all shells
            if len(no_dirs_pershell) == 1:
                no_dirs_pershell = no_dirs_pershell * len(bvals_nom)
    
            # Create nominal bvals
            bvals_nom_t = []
            bvals_nom_t = [value for value, count in zip(bvals_nom, no_dirs_pershell) for _ in range(count)]
            bvals_nom_t = np.concatenate(
                (np.zeros((no_b0, 1)), np.array(bvals_nom_t).reshape(-1, 1))).T
          
            # Format bmatrix
            bmatrix             = np.array(bmatrix).reshape(dims_bmat[1]*dims_bmat[2], dims_bmat[0], order='F')
            #bmatrix             = np.tile(bmatrix, no_rep)
            
            # Format effective bvals
            bvals_eff           = np.tile(np.array(bvals_eff)[:,np.newaxis].T, no_rep)
            
            # Save all files
            np.savetxt(bids_strc.get_path('bvalsEff.txt'), bvals_eff, delimiter=' ', fmt='%.1f')
            np.savetxt(bids_strc.get_path('bvalsNom.txt'), bvals_nom_t, delimiter=' ', fmt='%.1f')
            np.savetxt(bids_strc.get_path('bvecs.txt'), dirs, delimiter=' ', fmt='%.16f')
            np.savetxt(bids_strc.get_path('bmatrix.txt'), bmatrix, delimiter=' ', fmt='%.16f')
            
        else:
            raise ValueError('ParaVision version not recognized!')
    
    elif acqp == 'STE':

        bvals_nom  =  np.loadtxt(os.path.join(cfg['common_folder'],'STE_bvalsNom.txt'))
        bvals_eff  =  np.loadtxt(os.path.join(cfg['common_folder'],'STE_bvalsEff.txt'))
        bvecs_fake =  np.loadtxt(os.path.join(cfg['common_folder'],'STE_bvecs_fake.txt'))


        np.savetxt(bids_strc.get_path('bvalsNom.txt'), bvals_nom[np.newaxis, :], delimiter=' ', fmt='%.1f')
        np.savetxt(bids_strc.get_path('bvalsEff.txt'), bvals_eff[np.newaxis, :], delimiter=' ', fmt='%.1f')
        np.savetxt(bids_strc.get_path('bvecs_fake.txt'), bvecs_fake, delimiter=' ', fmt='%.1f')

def order_bvals(bvals):
      
    # 1. Create groups
    blocks = []
    start_idx = 0
    for value, group in groupby(enumerate(bvals), key=lambda x: x[1]):
        group = list(group)
        indices = [i for i, _ in group]
        blocks.append({
            'bval': value,
            'indices': indices,
            'values': [bvals[i] for i in indices]
        })
    
    # 2: Sort blocks by b-value
    sorted_blocks = sorted(blocks, key=lambda b: b['bval'])
    
    # 3: Reassemble
    sorted_bvals = []
    sorted_indices = []
    
    for block in sorted_blocks:
        sorted_bvals.extend(block['values'])
        sorted_indices.extend(block['indices'])

    return sorted_bvals, sorted_indices

##### IMAGE OPERATIONS - HANDLING #####

def binary_op(input_path1, input_path2, operation, output_path, cfg):

    exe = os.path.join(cfg["fsl_path"], "fslmaths")
    call = [exe,
            f'{input_path1}',
            f'{operation}',
            f'{input_path2}'
            f' {output_path}']

    print(' '.join(call))
    os.system(' '.join(call))


def image_operations(input_image1, input_image2, operation, output_image, cfg):

    # to choose between 'm' - multiply
    #                   '+' - add
    #                   '-' - substract
    #                   '/' - divide
    #                   '^2' - power

    nii_1_shape = nib.load(input_image1).shape
    nii_2_shape = nib.load(input_image2).shape

    if len(nii_1_shape) != len(nii_2_shape):
        raise ValueError('Images need to have the same shape!')
    else:
        nii_ndims = len(nii_1_shape)

    exe = os.path.join(cfg["ants_path"], "ImageMath")
    call = [exe,
            f'{nii_ndims}',
            f'{output_image}',
            f'{operation}',
            f'{input_image1}',
            f'{input_image2}']

    os.system(' '.join(call))


def erode_im(input_path, output_path, sigma, cfg):

    nii_shape = nib.load(input_path).shape
    nii_ndims = len(nii_shape)
    
    exe = os.path.join(cfg["ants_path"], "ImageMath")
    call = [exe,
            f'{nii_ndims}',
            f'{output_path}',
            f'ME',
            f'{input_path}',
            f'{sigma}']

    os.system(' '.join(call))

def erode_im_fsl(input_path, output_path, cfg):

    exe = os.path.join(cfg["fsl_path"], "fslmaths")
    call = [exe,
            f'{input_path}',
            f'-ero -ero -ero -ero',
            f'{output_path}']

    os.system(' '.join(call))
  
    
def fsl_mult(input_path1, input_path2, output_path, cfg):
    
    exe = os.path.join(cfg["fsl_path"], "fslmaths")
    call = [exe,
            f'{input_path1}',
            f'-mul {input_path2}',
            f'{output_path}']
    os.system(' '.join(call))
    
def dilate_im(input_path, output_path, sigma, cfg):

    nii_shape = nib.load(input_path).shape
    nii_ndims = len(nii_shape)

    exe = os.path.join(cfg["ants_path"], "ImageMath")
    call = [exe,
            f'{nii_ndims}',
            f'{output_path}',
            f'MD',
            f'{input_path}',
            f'{sigma}']

    os.system(' '.join(call))


def disassemble_4D(input_path, output_prefix, cfg):

    exe = os.path.join(cfg["ants_path"], "ImageMath")
    call = [exe,
            f'ImageMath 4',
            f'{output_prefix}',
            f'TimeSeriesDisassemble',
            f'{input_path}']

    os.system(' '.join(call))


def filter_clusters_by_size(input_path, output_path, min_size):
    # Load the binary image
    img = nib.load(input_path)
    data = img.get_fdata()

    # Label connected clusters of 1s
    labeled_data, num_features = label(data)

    # Find objects in the labeled array
    objects = find_objects(labeled_data)

    # Initialize an empty array to store the filtered data
    filtered_data = np.zeros_like(data)

    # Loop through each object and check its size
    for i, obj in enumerate(objects):
        cluster = (labeled_data[obj] == (i + 1))
        cluster_size = cluster.sum()

        # If the cluster size is greater than or equal to the min_size, keep it
        if cluster_size >= min_size:
            filtered_data[obj][cluster] = 1

    # Create a new Nifti1Image
    filtered_img = nib.Nifti1Image(filtered_data, img.affine, img.header)

    # Save the new image
    nib.save(filtered_img, output_path)


def threshold_image(input_image, output_image, thr_low, thr_high, cfg):  # rita

    nii_shape = nib.load(input_image).shape
    nii_ndims = len(nii_shape)
    
    exe = os.path.join(cfg["ants_path"], "ThresholdImage")
    call = [exe,
            f'{nii_ndims}',
            f'{input_image}',
            f'{output_image}',
            f'{thr_low}',
            f'{thr_high}']

    os.system(' '.join(call))


def dilate_image(input_image, output_image, cfg):

    exe = os.path.join(cfg["fsl_path"], "fslmaths")
    call = [exe,
            f'{input_image}',
            f'-dilD',
            # add -kernel flag in order to modify the box kernel / by default is 3x3x3
            f'{output_image}']

    os.system(' '.join(call))


def make_mask(input_path, output_path, val, cfg):
    
    exe = os.path.join(cfg["fsl_path"], "fslmaths")
    call = [exe,
            f'{input_path}',
            f'-thr {val} -bin',
            f'{output_path}']

    os.system(' '.join(call))
    
    # Fill in holes by creating mask
    call = [exe,
            f'{output_path} -fillh'  ,
            f'{output_path}']
    
    print(' '.join(call))
    os.system(' '.join(call))

def fsl_reorient(input_path, cfg):
    
    exe = os.path.join(cfg["fsl_path"], "fslreorient2std")
    call = [exe,
            f'{input_path}',
            f'{input_path}']

    os.system(' '.join(call))

def multiply_by_mask(input_path, output_path, mask_path, cfg):
    
    create_directory(output_path)
    if input_path.endswith('.nii.gz'):
        output_path = os.path.join(output_path,os.path.basename(input_path).replace('.nii.gz', '_masked.nii.gz'))
    elif input_path.endswith('.nii'):
        output_path = os.path.join(output_path,os.path.basename(input_path).replace('.nii', '_masked.nii.gz'))
       
    exe = os.path.join(cfg["fsl_path"], "fslmaths")
    call = [exe,
            f'{input_path}',
            f'-mul {mask_path}',
            f'{output_path}']

    os.system(' '.join(call))

def create_countour(mask, cfg):
    
    dilate_im(mask, mask.replace('.nii.gz','_dil.nii.gz'), '1')

    # make countour mask
    countour_path = mask.replace('.nii.gz', '_bo_contour.nii.gz')
    exe = os.path.join(cfg["fsl_path"], "fslmaths")
    call = [exe,
            f"{mask.replace('.nii.gz','_dil.nii.gz')}",
            f'-add',
            f'{mask}',
            f'-uthr 1',
            f"{mask.replace('.nii.gz', '_contour.nii.gz')}"]
    print(' '.join(call))
    os.system(' '.join(call))
               
    
def create_inverse_mask(mask_path, brain_mask, output_path, cfg):
    
    create_directory(output_path)
    if mask_path.endswith('.nii.gz'):
        output_path = os.path.join(output_path,os.path.basename(mask_path).replace('_mask.nii.gz', '_inv_mask.nii.gz'))
    elif mask_path.endswith('.nii'):
        output_path = os.path.join(output_path,os.path.basename(mask_path).replace('_mask.nii', '_inv_mask.nii.gz'))
     
    exe = os.path.join(cfg["fsl_path"], "fslmaths")
    call = [exe,
            f'{brain_mask}',
            f'-sub {mask_path}',
            f'{output_path}']

    os.system(' '.join(call))
    
##### IMAGE OPERATIONS - PROCESSING #####

def extract_b0(dwi_path, bvec_path, bval_path, output_path, cfg):

    for ii in range(len(dwi_path)):

        dwi = dwi_path[ii]
        bvec = bvec_path[ii]
        bval = bval_path[ii]
        output = output_path[ii]

        exe = os.path.join(cfg["mrtrix_path"], "dwiextract")
        call = [exe,
                f'-bzero',
                f'-fslgrad {bvec} {bval}',
                f'{dwi}',
                f'{output}',
                f'-force']

        os.system(' '.join(call))

def extract_vols(dwi_path, outputpath, volstart, volend, cfg):
    
    exe = os.path.join(cfg["fsl_path"], "fslroi")
    call = [exe,
            f'{dwi_path}', 
            f'{outputpath}',
            f'{volstart} {volend}']
    
    os.system(' '.join(call))



def dwi_extract(old_dataset, new_dataset, bvals_list, cfg):

    old_dwi = old_dataset["dwi"]
    old_bvals = old_dataset["bvals"]
    old_bvecs = old_dataset["bvecs"]
    new_dwi = new_dataset["dwi"]
    new_bvals = new_dataset["bvals"]
    new_bvecs = new_dataset["bvecs"]

    exe = os.path.join(cfg["mrtrix_path"], "dwiextract")
    call = [exe,
            f'{old_dwi}',
            f'{new_dwi}',
            f'-shells {bvals_list}',
            f'-fslgrad {old_bvecs} {old_bvals}',
            f'-export_grad_fsl {new_bvecs} {new_bvals} -force']

    print(' '.join(call))
    os.system(' '.join(call))
    


def denoise_vols_default_kernel(input_path, output_path, noise_path, cfg):
    """
    Function that denoises data with mrtrix
   
    Args:
        input_path (str)       : path where the original data is
        output_path (str)      : path where to save the denoised data
        noise_path  (str)      : path where to save the sigma map generated
        cfg                    : config structure
      
    Returns:
        none
    """
    
    # denoise data
    exe = os.path.join(cfg["mrtrix_path"], "dwidenoise")
    call = [exe,
            f'{input_path}',
            f'{output_path}',
            f'-noise {noise_path}',
            f'-debug',
            f'-force -info']
    os.system(' '.join(call))

    # calculate residuals
    res_path = output_path.replace('.nii.gz', '_res.nii.gz')
    exe = os.path.join(cfg["mrtrix_path"], "mrcalc")
    call = [exe,
            f'{input_path}',
            f'{output_path}',
            f'-subtract',
            f'{res_path} -force']
    os.system(' '.join(call))
        
    # calculate sigma map by hand, different definition as implemented in mrtrix
    sigma_path  = output_path.replace('.nii.gz','_sigma2.nii.gz')
    res         = nib.load(res_path).get_fdata()
    template    = nib.load(res_path)
    sigma       = np.std(res,3)
    sigma_img   = nib.Nifti1Image(sigma, affine=template.affine, header=template.header)
    nib.save(sigma_img,  sigma_path) 
    

def denoise_designer(input_path, bvecs, bvals, output_path, data_path, algorithm_name, cfg):
    """
    Function that denoises data in Designer Toolbox implemented in Docker
   
    Args:
        input_path (str)       : path where the original data is
        bvecs (str)            : path to the bvecs file
        bvals (str)            : path to the bvals file
        output_path (str)      : path where to save the denoised data
        data_path  (str)       : path to the main folder of analysis to mount the data folder in Docker
        algorithm_name (str)   : name of the algortihm to use for the denoising. 
                                    Options are: veraart, jespersen
            
    Returns:
        none
    """
     
    # calculate kernel size
    num_vols  = len(read_numeric_txt(bvals).T)
    N = math.ceil(num_vols ** (1/3))  
    if N % 2 == 0:
        N += 1  
        
    # convert to mif
    nifti_to_mif(input_path, bvecs, bvals, input_path.replace('.nii.gz','.mif'), cfg)

    # run denoising
    docker_path  = '/data'
    input_path   = input_path.replace(data_path,docker_path)
    input_path   = input_path.replace('.nii.gz','.mif')
    output_path  = output_path.replace(data_path,docker_path)
    output_path  = output_path.replace('.nii.gz','.mif')

    call = [f'docker run -v {data_path}:/data nyudiffusionmri/designer2:v2.0.13 designer -denoise',
            f'{input_path}',
            f'{output_path} -pf 0.75 -pe_dir i -algorithm {algorithm_name} -extent {N},{N},{N} -debug']
    
    os.system(' '.join(call))
    print(' '.join(call))

    # convert back to nii.gz 
    output_path  = output_path.replace(docker_path,data_path)
    nifti_to_mif(output_path, output_path.replace('.mif','.bvec'), output_path.replace('.mif','.bval'), output_path.replace('.mif','.nii.gz'), cfg)
    input_path   = input_path.replace(docker_path,data_path)
    input_path   = input_path.replace('.mif','.nii.gz')
    output_path  = output_path.replace('.mif','.nii.gz')
    # call = [f'flirt',
    #      f'-in  {output_path}',
    #      f'-ref {input_path}',
    #      f'-out {output_path}',
    #      f'-applyxfm -usesqform']
    # os.system(' '.join(call))

    # calculate residuals
    res_path = output_path.replace('.nii.gz','_res.nii.gz')
    exe = os.path.join(cfg["mrtrix_path"], "mrcalc")
    call     = [exe,
            f'{input_path}',
            f'{output_path}',
            f'-subtract',
            f'{res_path} -force']
    os.system(' '.join(call))
    
    # calculate sigma map
    sigma_path  = output_path.replace('.nii.gz','_sigma.nii.gz')
    res         = nib.load(res_path).get_fdata()
    template    = nib.load(res_path)
    sigma       = np.std(res,3)
    sigma_img   = nib.Nifti1Image(sigma, affine=template.affine, header=template.header)
    nib.save(sigma_img,  sigma_path) 

    
def denoise_matlab_NORDIC(input_path, output_path, delta_path, code_path, cfg):
    """
    Function that denoises data in matlab

    Args:
        input_path (str)       : path where the original data is 
        out_path (str)         : path where to save the denoised data. 
        delta_path (str)       : path to where the diffusion times associated with each volume in input_path is
        code_path  (str)       : path to where the matlab code that does the denoising is.
        cfg                    : config structure
        
    Returns:
        none
    """

    toolbox_path = os.path.join(cfg['toolboxes'], 'NORDIC_denoising')
    
    # output
    output_filename = os.path.basename(output_path)
    output_filename = output_filename.replace(".nii.gz", "")
    output_filedir  = os.path.dirname(output_path)

    matlab_cmd = (
        "try; "
         f"addpath('{toolbox_path}'); "
        "clear ARG; "
        "ARG.magnitude_only = 1; "
        "ARG.write_gzipped_niftis = 1; "
         f"ARG.DIROUT = '{output_filedir}/'; " # for some stupid reason NORDIC needs this path to end with /
        f"NIFTI_NORDIC('{input_path}', '{input_path}', '{output_filename}', ARG); "
        "catch ME, disp(ME.message); exit(1); "
        "end; exit(0);"
    )
    cmd = [
        "matlab", "-nodisplay", "-nosplash", "-nodesktop",
        "-r", matlab_cmd
    ]

    subprocess.run(cmd)
    
    # calculate residuals
    res_path = output_path.replace('.nii.gz','_res.nii.gz')
    exe = os.path.join(cfg["mrtrix_path"], "mrcalc")
    call     = [exe,
            f'{input_path}',
            f'{output_path}',
            f'-subtract',
            f'{res_path} -force']
    os.system(' '.join(call))
    
    # calculate sigma map by hand
    sigma_path  = output_path.replace('.nii.gz','_sigma.nii.gz')
    res         = nib.load(res_path).get_fdata()
    template    = nib.load(res_path)
    sigma       = np.std(res,3)
    sigma_img   = nib.Nifti1Image(sigma, affine=template.affine, header=template.header)
    nib.save(sigma_img,  sigma_path) 
    

def denoise_matlab(input_path, output_path, delta_path, code_path, cfg, dn_type):
    """
    Function that denoises data in matlab

    Args:
        input_path (str)       : path where the original data is 
        out_path (str)         : path where to save the denoised data. 
        delta_path (str)       : path to where the diffusion times associated with each volume in input_path is
        code_path  (str)       : path to where the matlab code that does the denoising is.
        cfg                    : config structure
        dn_type(str)           : string referring to the type of denoising. Options are: MPPCA, tMPPCA-4D, tMPPCA-5D
        
    Returns:
        none
    """

    toolbox_path = cfg['toolboxes']
    
    # unzip file because spm doesn't like .gz
    input_path = gunzip_file(input_path)

    # replace .nii.gz by just .nii
    output_path = output_path.replace('.nii.gz','.nii')
    
    # read delta times and calculate how many volumes per diffusion time exists
    Deltas = read_numeric_txt(delta_path)
    unique_deltas, counts = np.unique(Deltas, return_counts=True)

    # calculate kernel size
    num_vols  = nib.load(input_path).shape[-1]
    #num_vols = counts[0]
    N = math.ceil(num_vols ** (1/3))  
    if N % 2 == 0:
        N += 1  
     
    # matlab command
    matlab_cmd = (
        f"try, "
        f"addpath('{code_path}'); "
        f"denoise_in_matlab('{input_path}', '{output_path}' ,'{counts}','{N}', '{toolbox_path}', '{dn_type}'); "
        f"catch, exit(1), end, exit(0)"
    )
    cmd = [
        "matlab", "-nodisplay", "-nosplash", "-nodesktop",
        "-r", matlab_cmd
    ]

    subprocess.run(cmd)

    # gzip file
    output_path = gzip_file(output_path)
    
    # calculate residuals
    input_path = input_path.replace('.nii','.nii.gz')
    res_path = output_path.replace('.nii.gz','_res.nii.gz')
    exe = os.path.join(cfg["mrtrix_path"], "mrcalc")
    call     = [exe,
            f'{input_path}',
            f'{output_path}',
            f'-subtract',
            f'{res_path} -force']
    os.system(' '.join(call))
    
    gzip_file(output_path.replace('.nii.gz','_sigma.nii'))
    
    # calculate sigma map
    # sigma_path  = output_path.replace('.nii.gz','_sigma.nii.gz')
    # res         = nib.load(res_path).get_fdata()
    # template    = nib.load(res_path)
    # sigma       = np.std(res,3)
    # sigma_img   = nib.Nifti1Image(sigma, affine=template.affine, header=template.header)
    # nib.save(sigma_img,  sigma_path) 

# def denoise_vols(input_path, kernel_size, ouput_path, noise_path):

#     call = [f'dwidenoise',
#             f'{input_path}',
#             f'{ouput_path}',
#             f'-extent {kernel_size}',
#             f'-noise {noise_path}',
#             f'-debug',
#             f'-force']
#     os.system(' '.join(call))


# def denoise_vols_mask(input_path, kernel_size, mask_path, ouput_path, noise_path):

#     call = [f'dwidenoise',
#             f'{input_path}',
#             f'{ouput_path}',
#             f'-extent {kernel_size}',
#             f'-noise {noise_path}',
#             f'-mask {mask_path}',
#             f'-debug',
#             f'-force']

#     os.system(' '.join(call))


def denoise_img(input_path, dim, ouput_path,cfg):
    """
    Function that denoises data with Ants tools
   
    Args:
        input_path (str)       : path where the original data is
        dim (int)              : dimension of the dataset
        output_path (str)      : path where to save the denoised data
      
    Returns:
        none
    """
    
    exe = os.path.join(cfg["ants_path"], "DenoiseImage")
    call = [exe,
            f'-d {dim}',
            f'-i {input_path}',
            f'-o {ouput_path}',
            f'-v 1']

    os.system(' '.join(call))


def extract_bvals(dwi_path, bval_nom_path, bval_eff_path, bvec_path, bvals_list, output_pref):

    bval_nom_name = get_filename(bval_nom_path)
    bval_eff_name = get_filename(bval_eff_path)
    bvec_name = get_filename(bvec_path)

    # Extract dwi's corresponding to bvals_list
    dwi_extract(dwi_path, bval_nom_path, bvec_path, bvals_list, output_pref)

    bvals_nom = read_numeric_txt(bval_nom_path)
    bvals_eff = read_numeric_txt(bval_eff_path)

    # Extract b-vals nom and eff corresponding to bvals_list
    bvals_thr = list(map(int, bvals_list.split(',')))
    idx_bvals_thr = np.where(np.isin(bvals_nom.flatten(), bvals_thr))

    bvals_nom_thr = [bvals_nom.flatten()[index]
                     for index in list(idx_bvals_thr)]
    bvals_eff_thr = [bvals_eff.flatten()[index]
                     for index in list(idx_bvals_thr)]

    flat_bvals_nom_thr = np.concatenate(bvals_nom_thr).tolist() if isinstance(
        bvals_nom_thr[0], np.ndarray) else bvals_nom_thr
    write_txt(list(flat_bvals_nom_thr), bval_nom_path.replace(
        bval_nom_name, bval_nom_name + output_pref), 'w')

    flat_bvals_eff_thr = np.concatenate(bvals_eff_thr).tolist() if isinstance(
        bvals_eff_thr[0], np.ndarray) else bvals_eff_thr
    write_txt(list(flat_bvals_eff_thr), bval_eff_path.replace(
        bval_eff_name, bval_eff_name + output_pref), 'w')

    with open(bvec_path, 'r') as file:
        lines = file.readlines()

    # Extract columns based on indices
    extracted_lines = []
    for line in lines:
        # Split the line into columns (assuming space-separated values)
        columns = line.strip().split()
        # Extract the desired columns
        extracted_columns = [columns[i] for i in list(idx_bvals_thr[0])]
        # Join the extracted columns back into a string
        extracted_line = ' '.join(extracted_columns)
        extracted_lines.append(extracted_line)

    # Write the extracted lines to the output file
    with open(bvec_path.replace(bvec_name, bvec_name + output_pref), 'w') as file:
        for line in extracted_lines:
            file.write(line + '\n')
            
# def estim_DTI_DKI_designer(input_mif, 
#                            mask_path, output_path, data_path):

#     call = [f'docker run -v {data_path}:/data nyudiffusionmri/designer2:v2.0.10 tmi -SMI',
#             f'{input_mif}',
#             f'{output_path}',
#             f'-mask {mask_path}']

#     print(' '.join(call))
#     os.system(' '.join(call))

# def mdm_matlab(bids_LTE, bids_STE, bids_STE_reg, header, output_path, code_path, toolbox_path, low_b=False):
#     """
#     Function that computes micro FA in matlab

#     Args:
#         bids_LTE (str)         : BIDS structure of LTE data (linear tensor encoding)
#         bids_STE (str)         : BIDS structure of STE data (spherical tensor encoding)
#         bids_STE_reg (str)     : BIDS structure of STE data registered to the LTE space (spherical tensor encoding)
#         header (str)           : path to mask file
#         output_path (str)      : path where to save the computed measures. 
#         code_path  (str)       : path to where the matlab tollboxes for computing micro FA are.
#         toolbox_path  (str)    : path to where the toolboxes for matlab are. 
#                     They are added to the path in Matlab directly
#         low_b  (Bollean)       : True if the user wnats to use only the low bvalues to compute the microFa.
#                     Useful for regions like the CSF where at high b-values the noise floor is reached and then microFA doesn't make sense

#     Returns:
#         none
#     """
    
#     # unzip file because spm doesn't like .gz
#     STE_path  = bids_STE_reg.get_path('STE_in_LTE_dn_gc_topup.nii.gz')
#     LTE_path  = get_file_in_folder(bids_LTE,'*dwi_dn_gc_ec.nii.gz')
#     LTE_path = gunzip_file(LTE_path)
#     STE_path = gunzip_file(STE_path)

#     # replace .nii.gz by just .nii
#     output_path = output_path.replace('.nii.gz','.nii')
#     create_directory(output_path)
    
#     # prepare files
#     copy_files([LTE_path, STE_path, header], [os.path.join(output_path,'LTE.nii'),os.path.join(output_path,'STE.nii'),os.path.join(output_path,'mask.nii')])
#     copy_files([get_file_in_folder(bids_LTE,'*bvalsNom.txt'), get_file_in_folder(bids_LTE,'*bvecsRotated.txt')], [os.path.join(output_path,'LTE.bval'),os.path.join(output_path,'LTE.bvec')])
#     copy_files([bids_STE.get_path('bvalsNom.txt'), bids_STE.get_path('bvecs_fake.txt')], [os.path.join(output_path,'STE.bval'),os.path.join(output_path,'STE.bvec')])

#     if low_b==True:
#         # extract low b vals
#         bvals_LTE = read_numeric_txt(os.path.join(output_path,'LTE.bval'))
#         desiredbvals = np.unique(bvals_LTE[bvals_LTE<=2000])
#         old_dataset = {"dwi":  os.path.join(output_path,'LTE.nii'), 
#                         "bvals": os.path.join(output_path,'LTE.bval'),
#                         "bvecs": os.path.join(output_path,'LTE.bvec')}
#         new_dataset = {"dwi":   os.path.join(output_path,'LTE.nii'), 
#                         "bvals": os.path.join(output_path,'LTE.bval'),
#                         "bvecs": os.path.join(output_path,'LTE.bvec')}              
#         dwi_extract(old_dataset, new_dataset,','.join(map(str, desiredbvals)))
    
#     # new paths
#     LTE_path = os.path.join(output_path,'LTE.nii')
#     STE_path = os.path.join(output_path,'STE.nii')

#     # matlab command
#     matlab_cmd = (
#         f"try, "
#         f"addpath('{code_path}'); "
#         f"calculate_microFA('{LTE_path}', '{STE_path}' ,'{header}','{output_path}', '{toolbox_path}'); "
#         f"catch, exit(1), end, exit(0)"
#     )
#     cmd = [
#         "matlab", "-nodisplay", "-nosplash", "-nodesktop",
#         "-r", matlab_cmd
#     ]
    
#     subprocess.run(cmd)



def extract_dwi_by_bval(nifti_in, bvals, nifti_out, keep_mask):
    img = nib.load(nifti_in)
    data = img.get_fdata(dtype=np.float32)          # (X,Y,Z,N)
    data2 = data[:,:,:, keep_mask]                    # slice volumes

    out_img = nib.Nifti1Image(data2, img.affine, img.header)
    out_img.header.set_data_shape(data2.shape)      # keep header consistent
    nib.save(out_img, nifti_out)


def compute_micro_FA(bids_LTE, bids_STE, mask_path, output_path):
    """
    Function that computes micro FA in python. Based in mdm_matlab
    It computes micro FA from LTE and STE data only.
    Other waveforms are not accounted for!

    Args:
        bids_LTE (str)         : BIDS structure of LTE data (linear tensor encoding)
        bids_STE (str)         : BIDS structure of STE data (spherical tensor encoding)
        mask_path (str)        : path to mask file
        output_path (str)      : path where to save the computed measures. 

    Returns:
        none
    """
    
    # create output folder
    create_directory(output_path)
    
    # prepare files
    copy_files([get_file_in_folder(bids_LTE,'*pwd_avg.nii.gz'),  get_file_in_folder(bids_STE,'*pwd_avg.nii.gz')], [os.path.join(output_path,'LTE.nii.gz'),os.path.join(output_path,'STE.nii.gz')])
    copy_files([get_file_in_folder(bids_LTE,'*bvalsNom_avg.txt'),get_file_in_folder(bids_STE,'*bvalsNom_avg.txt')], [os.path.join(output_path,'LTE_bval.txt'),os.path.join(output_path,'STE_bval.txt')])
    copy_files([get_file_in_folder(bids_LTE,'*bvalsNom_shells.txt'),get_file_in_folder(bids_STE,'*bvalsNom_shells.txt')], [os.path.join(output_path,'LTE_shell.txt'),os.path.join(output_path,'STE_shell.txt')])
    copy_files([mask_path], [os.path.join(output_path,'mask.nii.gz')])

    # extract low b vals of LTE
    bvals_LTE = read_numeric_txt(os.path.join(output_path,'LTE_bval.txt'))[0].astype(float)
    keep = (bvals_LTE <= 2)
    extract_dwi_by_bval(
        nifti_in=os.path.join(output_path, "LTE.nii.gz"),
        bvals=bvals_LTE,
        nifti_out=os.path.join(output_path, "LTE.nii.gz"),
        keep_mask=keep
    )
    write_txt(bvals_LTE[keep], os.path.join(output_path, "LTE_bval.txt"),'w')
    pa_w   = read_numeric_txt( os.path.join(output_path,'LTE_shell.txt'))[0]
    write_txt(pa_w[keep], os.path.join(output_path, "LTE_shell.txt"),'w')

    # concatenate LTE and STE (first LTE then STE)
    concat_niftis([os.path.join(output_path,'LTE.nii.gz'),
                   os.path.join(output_path,'STE.nii.gz')], 
                   os.path.join(output_path,'merged_dwi.nii.gz'), 'all')
    concat_files([os.path.join(output_path,'LTE_shell.txt'),
                  os.path.join(output_path,'STE_shell.txt')],
                  os.path.join(output_path,'merged_pa_w.txt'))
    concat_files([os.path.join(output_path, 'LTE_bval.txt'),
                  os.path.join(output_path, 'STE_bval.txt')],
                  os.path.join(output_path,'merged_bvals.txt'))
    
    # load bvals
    b_vals = read_numeric_txt( os.path.join(output_path,'merged_bvals.txt'))
    b_vals = np.asarray(b_vals, float).ravel()
    
    # number of bvals per shell
    pa_w   = read_numeric_txt( os.path.join(output_path,'merged_pa_w.txt')) 
    pa_w   = np.asarray(pa_w, float).ravel()

    # number of bvals LTE and STE
    n_LTE = len(open(os.path.join(output_path, 'LTE_shell.txt')).read().split()) 
    n_STE = len(open(os.path.join(output_path, 'STE_shell.txt')).read().split()) 

    # indices making the map between each volume and STE or LTE data
    s_ind = np.r_[1* np.ones(n_LTE),2 * np.ones(n_STE)] 
    
    # b_eta and b_delta params that define the encoding space (the definition 
            # is a bit confusing but in agreement with outputed from matlab)
    b_delta_sphere = 0
    b_delta_stick = 1
    b_eta         = np.r_[0  * np.ones(n_LTE), 0 * np.ones(n_STE)] 
    b_delta       = np.r_[b_delta_stick  * np.ones(n_LTE),
                          b_delta_sphere * np.ones(n_STE)] 
    
    # load image
    img = nib.load(os.path.join(output_path,'merged_dwi.nii.gz'))
    I = img.get_fdata().astype(float)
    
    # mask
    mask = nib.load(os.path.join(output_path,'mask.nii.gz')).get_fdata()
    idx = np.argwhere(mask != 0) 
    
    # initiate output images
    names = ["S0", "MD", "MKiso", "MKaniso", "Uaniso", "Uiso", "Vl", "microFA"]
    vols = {n: np.full(I.shape[:3], np.nan, dtype=float) for n in names}
    
    # fit each voxel
    print('Fitting gamma distribution, it takes a while ...')
    for i, j, k in idx:
       
        # estimate parameters
        signal    = I[i, j, k, :]
        estimates_in_SI, unit_to_SI = fit_voxel(signal, b_vals, b_eta, b_delta, pa_w, s_ind)

        # compute variables
        s0, d_iso, mu2_iso, mu2_aniso = estimates_in_SI[:4]
        vols["S0"][i, j, k] = s0
        vols["MD"][i, j, k] = d_iso / unit_to_SI[1]
        denom = (d_iso**2) if d_iso != 0 else np.nan
        vols["MKiso"][i, j, k] = 3.0 * mu2_iso   / denom
        vols["MKaniso"][i, j, k] = 3.0 * mu2_aniso / denom
        vols["Uiso"][i, j, k]   = mu2_iso   / unit_to_SI[2]
        vols["Uaniso"][i, j, k] = mu2_aniso / unit_to_SI[3]
        Vl = 5/2 * mu2_aniso 
        vols["Vl"][i, j, k] = Vl
        uFA = np.sqrt(3/2) * np.sqrt(Vl / (Vl + mu2_iso + d_iso**2)) if (Vl + mu2_iso + d_iso**2) > 0 else np.nan
        vols["microFA"][i, j, k] = uFA
    
    # save nifti            
    ref = nib.load(os.path.join(output_path,'mask.nii.gz'))
    affine = ref.affine
    header = ref.header.copy()
    header.set_data_dtype(np.float32)
    for name, vol in vols.items():
        img = nib.Nifti1Image(np.asarray(vol, np.float32), affine, header=header)
        out_path = os.path.join(output_path, f"{name}.nii.gz")
        nib.save(img, out_path)
                
def fit_voxel(signal, b_vals, b_eta, b_delta, pa_w, s_ind):
    from scipy.optimize import least_squares
 
    # ensure signal format
    y      = np.asarray(signal, float).ravel()

    # bvals in SI
    b_vals_in_SI = b_vals*1e9
    
    # bounds in SI: [s0, d_iso, mu2_iso, mu2_aniso, r_ste]  (r_ste unitless)
    smax     = float(np.max( y + np.finfo(float).eps))
    m_lb     = np.array([0.0, 1e-12, 1e-24, 1e-24, 0.5], dtype=float)
    m_ub     = np.array([10.0, 4e-9, 5e-18, 5e-18, 2.0], dtype=float)
    m_lb[0] *= smax
    m_ub[0] *= smax

    # mapping for conversion to SI
    unit_to_SI = np.array([smax, 1e-9, (1e-9)**2, (1e-9)**2, 1.0], float)
    def t2m(t):  # t -> SI
        return np.asarray(t, float) * unit_to_SI

    # weights 
    w = np.sqrt(pa_w / np.max(pa_w))
    w = np.asarray(w, float).ravel()
  
    # bounds in original units
    t_lb = m_lb / unit_to_SI
    t_ub = m_ub / unit_to_SI
    m0_SI = random_initial_guess_dtdgamma(
        signal=y,
        b_vals_SI=b_vals_in_SI,
        b_eta=b_eta,
        b_delta=b_delta,
        s_ind=s_ind,
        lb_SI=m_lb,
        ub_SI=m_ub,
        weight=w,
        iterations=50,   
    )
    
    # convert SI to t-space
    x0 = m0_SI / unit_to_SI

    # residuals 
    def residuals(t):
        m_SI = t2m(t)
        y_hat = dtd_gamma_1d_fit2data_py(m_SI, b=b_vals_in_SI, b_eta=b_eta, b_delta=b_delta, s_ind=s_ind)
        y_hat = np.asarray(y_hat, float).ravel()
        return (y_hat - y) * w
    
    # fit
    diff_step = np.sqrt(np.finfo(float).eps)
    out = least_squares(
        fun=residuals,                 
        x0=x0,
        bounds=(t_lb, t_ub),
        method="trf",                  # MATLAB trust-region-reflective
        jac="2-point",                 # MATLAB FiniteDifferenceType='forward'
        diff_step=diff_step,           # MATLAB FiniteDifferenceStepSize=sqrt(eps)
        ftol=1e-6,                     # MATLAB FunctionTolerance
        xtol=1e-6,                     # MATLAB StepTolerance
        gtol=1e-6,                     # MATLAB OptimalityTolerance
        max_nfev=1000,                 # MATLAB MaxFunctionEvaluations
        x_scale=np.ones_like(x0),      # MATLAB TypicalX = ones(...)
        verbose=0,                    
    )

    # return SI params 
    t_hat = out.x
    m_hat_SI = t2m(t_hat)
   

    return m_hat_SI, unit_to_SI  # [s0, d_iso, mu2_iso, mu2_aniso] in SI


def dtd_gamma_1d_fit2data_py(
    m: np.ndarray,
    b: np.ndarray,
    b_eta: np.ndarray,
    b_delta: np.ndarray,
    s_ind: np.ndarray,
):
    """
    Python equivalent of MATLAB dtd_gamma_1d_fit2data from 
    https://github.com/markus-nilsson/md-dmri/tree/master

    Args:
        m : (5,) array. [s0, d_iso, mu2_iso, mu2_aniso, s_w] in SI units.
                        s0        - signal at b0
                        d_iso     - isotropic diffusivity
                        mu2_iso   - μ2 from isotropic part
                        mu2_aniso - μ2 from anisotropic part
                        s_w       - value for scaling factor between 
                            the different types of acquisition (STE/LTE)  
        b : (N,) array. b-values (SI units, i.e., s/m^2).
        b_eta : (N,) array. b_eta per measurement (LTE=0, STE=0). Same size as b.
        b_delta : (N,) array. b_delta per measurement (LTE=1, STE=0). Same size as b.
        s_ind : (N,) int array. Series with the corresponding acquisition type for each element 
                                in the array: indices 1 (=LTE) and 2 (=STE). Same size as b.

    Returns:
        s : (N,) array. Predicted signal (real-valued).
    """
        
    m = np.asarray(m, float)
    b = np.asarray(b, float)
    b_eta   = np.asarray(b_eta, float)
    b_delta = np.asarray(b_delta, float)

    s0        = m[0]
    d_iso     = m[1]
    mu2_iso   = m[2]
    mu2_aniso = m[3]

    # Relative scaling factors across series: 
    rs = np.concatenate([np.array([1.0]), m[4:]])  
    s_ind = np.asarray(s_ind).astype(int).ravel()
    if rs.size > 1:
        s_ind = np.asarray(s_ind)
        sw = s0 * rs[s_ind - 1]   
    else:
        sw = s0 * np.ones_like(b)

    mu2 = mu2_iso + mu2_aniso * (b_delta**2) * ((b_eta**2 + 3.0) / 3.0)

    s = sw * (1.0 + b * mu2 / d_iso) ** (-(d_iso**2) / mu2)

    return np.real(s)

def random_initial_guess_dtdgamma(
    signal,
    b_vals_SI,
    b_eta,
    b_delta,
    s_ind,
    lb_SI,
    ub_SI,
    weight=None,
    iterations=100,
):
    """
    Random initial guess selector for dtd_gamma model (SI units) from
    https://github.com/markus-nilsson/md-dmri/tree/master
    """
    signal  = np.asarray(signal, float).ravel()
    b_vals  = np.asarray(b_vals_SI, float).ravel()
    b_eta   = np.asarray(b_eta,  float).ravel()
    b_delta = np.asarray(b_delta,  float).ravel()

    if weight is None:
        weight = np.ones_like(signal, float)
    else:
        weight = np.asarray(weight, float).ravel()

    thr = np.inf
    best = None

    for _ in range(iterations):
        m_rand = lb_SI + (ub_SI - lb_SI) * np.random.rand(*lb_SI.shape)

        s_rand = dtd_gamma_1d_fit2data_py(
            m_rand,
            b=b_vals,
            b_eta=b_eta,
            b_delta=b_delta,
            s_ind=s_ind,
        )
        s_rand = np.asarray(s_rand, float).ravel()

        r = np.sum(((signal - s_rand) * weight) ** 2)

        if r < thr:
            thr = r
            best = m_rand

    return best
   
def calculate_pwd_avg(dwi_path, bval_nom_path, bval_eff_path, output_path, diff_time):

    bvals_nom = read_numeric_txt(bval_nom_path)
    sorted_bvals_nom_unique, idx, sorted_counts = np.unique(bvals_nom, return_index=True, return_counts=True)
    #sorted_bvals_nom_unique = bvals_nom_unique[np.argsort(idx)]
   
    bvals_eff = read_numeric_txt(bval_eff_path)
    bvals_eff = bvals_eff.flatten()
    bval_eff_avg = []

    dwi_img = nib.load(dwi_path)
    dwi_array = dwi_img.get_fdata()

    dwi_name = get_filename(dwi_path)
    bval_nom_name = get_filename(bval_nom_path)
    bval_eff_name = get_filename(bval_eff_path)

    # Calculate average b0
    b0_idx = np.where(bvals_nom == 0.0)[1]
    b0_vols = dwi_array[:, :, :, b0_idx]
    b0_avg = np.mean(b0_vols, axis=3)

    dwi_avg = np.empty(tuple(
        dwi_img.shape[0:3]) + (len(sorted_bvals_nom_unique),))
    dwi_avg_norm = np.empty(tuple(
        dwi_img.shape[0:3]) + (len(sorted_bvals_nom_unique),))

    for ii in range(len(sorted_bvals_nom_unique)):

        bval_idx = np.where(bvals_nom == sorted_bvals_nom_unique[ii])[1]
        dwi_vols = dwi_array[:, :, :, bval_idx]
        dwi_temp = np.mean(dwi_vols, axis=3)
        dwi_temp_norm = dwi_temp / b0_avg
        
        dwi_avg[:, :, :, ii] = dwi_temp
        dwi_avg_norm[:, :, :, ii] = dwi_temp_norm

        bval_eff_avg.append(
            round(np.mean(bvals_eff[np.array(bval_idx)])/1e3, 2))

        array_to_nii(dwi_img, dwi_temp, os.path.join(output_path, dwi_name +
                     '_' + str(sorted_bvals_nom_unique[ii].astype(int)) + '.nii.gz'))
        array_to_nii(dwi_img, dwi_temp_norm, os.path.join(output_path, dwi_name +
                     '_norm_' + str(sorted_bvals_nom_unique[ii].astype(int)) + '.nii.gz'))

    array_to_nii(dwi_img, dwi_avg, os.path.join(
        output_path, dwi_name + '_pwd_avg.nii.gz'))
    array_to_nii(dwi_img, dwi_avg_norm, os.path.join(
        output_path, dwi_name + '_pwd_avg_norm.nii.gz'))
    
    write_txt(sorted_bvals_nom_unique / 1e3, os.path.join(output_path, bval_nom_name + '_avg.txt'), 'w')
    # write_txt([str(diff_time)] * len(sorted_bvals_nom_unique), os.path.join(output_path, 'td.txt'), 'w')
    write_txt(sorted_counts, os.path.join(output_path, bval_nom_name + '_shells.txt'), 'w')
    write_txt(bval_eff_avg, os.path.join(
        output_path, bval_eff_name + '_avg.txt'), 'w')


def N4_unbias(input_path, output_path, cfg):

    size = nib.load(input_path).shape
    dim = len(size)

    if dim == 2 or dim == 3 or dim == 4:
        exe = os.path.join(cfg["ants_path"], "N4BiasFieldCorrection")
        call = [exe,
                f'-d {dim} ',
                f'-i {input_path}',
                f'-o {output_path}',
                f'-c [50x50x50x50,0.00001]',
                f'-v 1']
        os.system(' '.join(call))

    else:
        raise ValueError('Image dimension not supported!')


def gibbs_corr(input_path, output_path, cfg):

    exe = os.path.join(cfg["mrtrix_path"], "mrdegibbs")
    call = [exe,
            f'{input_path}',
            f'{output_path}',
            f'-force']

    os.system(' '.join(call))


def make_avg(dim, input_path, output_path, cfg):  

    for ii in range(len(input_path)):

        input = input_path[ii]
        output = output_path[ii]

        exe = os.path.join(cfg["ants_path"], "antsMotionCorr")
        call = [exe,
                f'-d {dim}',
                f'-a {input}',
                f'-o {output}']

        os.system(' '.join(call))
        
def calc_snr(input_path1, input_path2, output_path, cfg):
    
    binary_op(input_path1, input_path2, '-div', output_path, cfg)

def brain_extract_BET(input_path, cfg):
    
    exe = os.path.join(cfg["fsl_path"], "bet")
    call = [exe,
            f'{input_path}',
            f'{input_path.replace(".nii.gz", "_brain.nii.gz")} -R -m']
    
    print(' '.join(call))
    os.system(' '.join(call))
    
def brain_extract_RATS(input_path, cfg):

    # Segment with RATS to create brain mask
    exe = os.path.join(cfg["rats_path"], "RATS_MM")
    call = [exe,
            f'{input_path}',
            f'{input_path.replace(".nii.gz", "_brain_mask.nii.gz")}',
            f'-t 1500']
    print(' '.join(call))
    os.system(' '.join(call))  
    
def brain_mask_refine(input_path,anat_thr, cfg):

    # Use brain mask to get just the T2w brain image
    binary_op(input_path,input_path.replace(".nii.gz", "_brain_mask.nii.gz"), '-mul', input_path.replace(".nii.gz", "_brain.nii.gz"), cfg)
    
    # Apply extra threshold on intensity
    exe = os.path.join(cfg["fsl_path"], "fslmaths")
    call = [exe,
            f'{input_path.replace(".nii.gz", "_brain.nii.gz")}',
            f'-thr {anat_thr}', # 4000, 2100
            f'{input_path.replace(".nii.gz", "_brain.nii.gz")}']
    
    print(' '.join(call))
    os.system(' '.join(call))
    
    # Fill in holes by creating mask
    call = [exe,
            f'{input_path.replace(".nii.gz", "_brain.nii.gz")} -fillh'  ,
            f'{input_path.replace(".nii.gz", "_brain_mask.nii.gz")}']
    
    print(' '.join(call))
    os.system(' '.join(call))
    
    # Extra clean of mask
    import scipy.ndimage as ndi

    img  = nib.load(input_path.replace(".nii.gz", "_brain_mask.nii.gz"))
    data = img.get_fdata() > 0  # ensure bool
    
    # 1) Slight erosion to break thin bridges
    eroded = ndi.binary_erosion(data, structure=np.ones((3,3,3)), iterations=2)
    
    # 2) Keep only largest connected component
    labeled, n = ndi.label(eroded)
    if n > 0:
        largest_label = np.argmax(np.bincount(labeled.flat)[1:]) + 1
        largest_component = (labeled == largest_label)
    else:
        largest_component = eroded
    
    # 3) Dilate back to recover a bit of size
    cleaned = ndi.binary_dilation(largest_component, structure=np.ones((3,3,3)), iterations=2)
    
    nib.save(
        nib.Nifti1Image(cleaned.astype(np.uint8), img.affine),
        input_path.replace(".nii.gz", "_brain_mask.nii.gz")
    )
    
    # Extra clean of mask
    # img     = nib.load(input_path.replace(".nii.gz", "_brain_mask.nii.gz"))
    # data    = img.get_fdata()
    # cleaned = binary_opening(data, structure=np.ones((3,3,3)))
    # labeled, n = label(cleaned) # Keep only the largest connected component
    # largest_component = (labeled == np.argmax(np.bincount(labeled.flat)[1:]) + 1)
    # nib.save(nib.Nifti1Image(largest_component.astype(np.uint8), img.affine), input_path.replace(".nii.gz", "_brain_mask.nii.gz"))
    
  
    # Use new clean brain mask (there is never too many masks xD ) to get just the T2w brain image
    binary_op(input_path,input_path.replace(".nii.gz", "_brain_mask.nii.gz"), '-mul', input_path.replace(".nii.gz", "_brain.nii.gz"),cfg)

    
def brain_mask_refine_organoids(input_path,anat_thr, cfg):

    # Use brain mask to get just the T2w brain image
    #binary_op(input_path,input_path.replace(".nii.gz", "_brain_mask.nii.gz"), '-mul', input_path.replace(".nii.gz", "_brain.nii.gz"), cfg)
    
    # Apply extra threshold on intensity
    exe = os.path.join(cfg["fsl_path"], "fslmaths")
    call = [exe,
            f'{input_path}',
            f'-thr {anat_thr}', # 4000, 2100
            f'{input_path.replace(".nii.gz", "_brain.nii.gz")}']
    
    print(' '.join(call))
    os.system(' '.join(call))
    
    exe = os.path.join(cfg["fsl_path"], "fslmaths")
    call = [exe,
            f'{input_path.replace(".nii.gz", "_brain.nii.gz")}',
            f'-thr {anat_thr} -bin', # 4000, 2100
            f'{input_path.replace(".nii.gz", "_brain_mask.nii.gz")}']
    
    print(' '.join(call))
    os.system(' '.join(call))
    
    # # Fill in holes by creating mask
    # call = [exe,
    #         f'{input_path.replace(".nii.gz", "_brain.nii.gz")} -fillh'  ,
    #         f'{input_path.replace(".nii.gz", "_brain_mask.nii.gz")}']
    
    # print(' '.join(call))
    # os.system(' '.join(call))
    
    # # Extra clean of mask
    # import scipy.ndimage as ndi

    # img  = nib.load(input_path.replace(".nii.gz", "_brain_mask.nii.gz"))
    # data = img.get_fdata() > 0  # ensure bool
    
    # # 1) Slight erosion to break thin bridges
    # eroded = ndi.binary_erosion(data, structure=np.ones((3,3,3)), iterations=2)
    
    # # 2) Keep only largest connected component
    # labeled, n = ndi.label(eroded)
    # if n > 0:
    #     largest_label = np.argmax(np.bincount(labeled.flat)[1:]) + 1
    #     largest_component = (labeled == largest_label)
    # else:
    #     largest_component = eroded
    
    # # 3) Dilate back to recover a bit of size
    # cleaned = ndi.binary_dilation(largest_component, structure=np.ones((3,3,3)), iterations=2)
    
    # nib.save(
    #     nib.Nifti1Image(cleaned.astype(np.uint8), img.affine),
    #     input_path.replace(".nii.gz", "_brain_mask.nii.gz")
    # )
    
  
  
    # Use new clean brain mask (there is never too many masks xD ) to get just the T2w brain image
    #binary_op(input_path,input_path.replace(".nii.gz", "_brain_mask.nii.gz"), '-mul', input_path.replace(".nii.gz", "_brain.nii.gz"),cfg)


def interactive_brain_mask_refine(anat_path, subj_data, cfg, thr_mask_col='acqType'):
    """
    Run brain_mask_refine and interactively ask the user to accept or change the threshold.
    Expects subj_data to have a column 'anat_thr' and an acquisition-type column (thr_mask_col).
    """
    # Get initial threshold for the anatomical image
    mask = subj_data[thr_mask_col].str.contains('T2W|T1W', case=False)
    anat_thr = float(subj_data.loc[mask, 'anat_thr'].iloc[0])
    
    # Copy files in case
    copy_files([anat_path.replace('.nii.gz','_brain_mask.nii.gz')],
               [anat_path.replace('.nii.gz','_brain_mask_bkp.nii.gz')])

    while True:
        print(f"\nRunning brain_mask_refine with threshold anat_thr={anat_thr} ...")
        
        # Get the original mask again
        copy_files([anat_path.replace('.nii.gz','_brain_mask_bkp.nii.gz')],
                   [anat_path.replace('.nii.gz','_brain_mask.nii.gz')])

        # Refine
        if cfg['subject_type']=='organoid':
            brain_mask_refine_organoids(anat_path, anat_thr, cfg)
        else:
            brain_mask_refine(anat_path, anat_thr, cfg)

        ans = input("\n >> Check the final brain mask image (with FSLEyes eg). Is the brain mask OK? [Y/n]: ").strip().lower()
        if ans in ("", "y", "yes"):
            # Save the final threshold back into subj_data
            subj_data.loc[mask, 'anat_thr'] = anat_thr
            print(f"Brain mask accepted. Final anat_thr set to {anat_thr}.")
            break

        # Ask for a new threshold
        while True:
            new_thr_str = input("Enter new threshold for brain mask: ").strip()
            try:
                anat_thr = float(new_thr_str)
                break
            except ValueError:
                print("Invalid value, please enter a numeric threshold.")
                
    return anat_thr


def brain_extract_organoids(input_path,val, cfg):
    
    #Make brain mask
    make_mask(input_path, input_path.replace(".nii.gz", "_brain_mask.nii.gz"), val)
    
    # Use brain mask to get just the T2w brain image
    binary_op(input_path,input_path.replace(".nii.gz", "_brain_mask.nii.gz"), '-mul', input_path.replace(".nii.gz", "_brain.nii.gz"), cfg)
    
    
def brain_extract_BREX(input_path,BREX_path):
    out_path = os.path.dirname(input_path)
    os.chdir(out_path)
    copy_files([os.path.join(BREX_path, 'atlasBREX.sh'),
                os.path.join(BREX_path, 'templates',
                             'mouse_T2w', 'b_mouse_T2w.nii.gz'),
                os.path.join(BREX_path, 'templates', 'mouse_T2w', 'nb_mouse_T2w.nii.gz')],
               [os.path.join(out_path, 'atlasBREX.sh'),
                os.path.join(out_path, 'b_mouse_T2w.nii.gz'),
                os.path.join(out_path, 'nb_mouse_T2w.nii.gz')])
   
    # antsreg(os.path.join(out_path, 'b_mouse_T2w.nii.gz'),input_path, input_path.replace('.nii.gz', 'template2anat'))
    # ants_apply_transforms([os.path.join(out_path, 'b_mouse_T2w.nii.gz'),os.path.join(out_path, 'nb_mouse_T2w.nii.gz')],
    #                               input_path,
    #                               [os.path.join(out_path, 'b_mouse_T2w_in_anat.nii.gz'),os.path.join(out_path, 'nb_mouse_T2w_in_anat.nii.gz')],
    #                               input_path.replace('.nii.gz', 'template2anat0GenericAffine.mat'),
    #                               input_path.replace('.nii.gz', 'template2anat1Warp.nii.gz'))                  
          

    call = [f'bash {os.path.join(out_path, "atlasBREX.sh")}',
            f'-b {os.path.join(out_path, "b_mouse_T2w.nii.gz")}',
            f'-nb {os.path.join(out_path, "nb_mouse_T2w.nii.gz")}',
            f'-h {input_path} -f 0.01 ']
    

    os.system(' '.join(call))
    
    remove_file(os.path.join(out_path, 'atlasBREX.sh'))
    remove_file(os.path.join(out_path, 'b_mouse_T2w.nii.gz'))
    remove_file(os.path.join(out_path, 'nb_mouse_T2w.nii.gz'))


def register_outputfits_to_anat(output_path, new_output_path,model,cfg, bids_strc_anat,bids_strc_prep):
     from .custom_functions_models import get_param_names_model
 
     # auxiliar function to register output model fits to anatomical space
     anat_format = cfg['anat_format']
     create_directory(new_output_path)
     patterns, lims, maximums = get_param_names_model(model,cfg['is_alive'])
    
     for filename in os.listdir(output_path):
         if  any(fnmatch.fnmatch(filename, pattern) for pattern in patterns):
             
             if cfg['subject_type']=='organoid':
                  # pad image temporarily for registration
                  pad_image(os.path.join(output_path, filename),os.path.join(output_path, filename))
                  pad_image(bids_strc_anat.get_path(f'{anat_format}_bc_brain.nii.gz'),bids_strc_anat.get_path(f'{anat_format}_bc_brain.nii.gz'))
    
                  # Apply inverse transform to put template anat in dwi
                  # ants_apply_transforms([os.path.join(output_path, filename)],  # input 
                  #                      bids_strc_anat.get_path(f'{anat_format}_bc_brain.nii.gz'), # reference
                  #                      [os.path.join(new_output_path,filename.replace('.nii','_padded.nii'))], # output
                  #                      [bids_strc_prep.get_path(f'dwiafterpreproc2{anat_format}0GenericAffine.mat'), 0], # transform 1
                  #                      bids_strc_prep.get_path(f'dwiafterpreproc2{anat_format}1Warp.nii.gz')) # transform 2
                  ants_apply_transforms_simple([os.path.join(output_path, filename)],  # input 
                                       bids_strc_anat.get_path(f'{anat_format}_bc_brain.nii.gz'), # reference
                                       [os.path.join(new_output_path,filename.replace('.nii','_padded.nii'))], # output
                                       [bids_strc_prep.get_path(f'dwiafterpreproc2{anat_format}0GenericAffine.mat'), 0],cfg) # transform 1
                            
    
                  # unpad the images previousy padded
                  unpad_image(os.path.join(output_path, filename),os.path.join(output_path, filename))
                  unpad_image(bids_strc_anat.get_path(f'{anat_format}_bc_brain.nii.gz'),bids_strc_anat.get_path(f'{anat_format}_bc_brain.nii.gz'))
                  unpad_image(os.path.join(new_output_path,filename.replace('.nii','_padded.nii')),os.path.join(new_output_path, filename))
                  remove_file(os.path.join(new_output_path,filename.replace('.nii','_padded.nii')))
                              
             else:
                 # Apply inverse transform to put template anat in dwi
                 ants_apply_transforms_simple([os.path.join(output_path, filename)],  # input 
                                      bids_strc_anat.get_path(f'{anat_format}_bc_brain.nii.gz'), # reference
                                      [os.path.join(new_output_path,filename)], # output
                                      [bids_strc_prep.get_path(f'dwiafterpreproc2{anat_format}0GenericAffine.mat'), 0],cfg) # transform 1
        
 
##### NIFTI HANDLE #####


def nifti_to_mif(nifti_path, bvecs_path, bvals_path, mif_path, cfg):

    exe = os.path.join(cfg["mrtrix_path"], "mrconvert")
    call = [exe,
            f'-fslgrad {bvecs_path} {bvals_path} ',
            f'{nifti_path} {mif_path} -force']

    os.system(' '.join(call))


def reorient_nifit(file_path, new_orient, cfg):


    if new_orient == 'x -z y':
        
        # Load data for later
        img = nib.load(file_path)
        affine = img.affine
        data = img.get_fdata()
        
        ## Delete header and swap dimensions ##
        exe = os.path.join(cfg["fsl_path"], "fslorient")
        call = [exe,
                f'-deleteorient {file_path}']
        os.system(' '.join(call))
        
        # the new direction x -z -y is found by trial and error to match the default MNI.
        # this will give a warning saying the L/R directions were flipped, but we will put them back later
        exe = os.path.join(cfg["fsl_path"], "fslswapdim")
        call = [exe,
                f'{file_path} x -z -y {file_path}']
        os.system(' '.join(call))
        
        exe = os.path.join(cfg["fsl_path"], "fslorient")
        call = [exe,
            f'-setqformcode 1 {file_path}']
        os.system(' '.join(call))
        
        ## Put back the header ##
        new_affine = affine.copy()
        
        # RL (x) stays the same as the initial file
        new_affine[0]=affine[0]
        
        # PA (y) is swapped with -SI (-z)
        new_affine[1]=-affine[2] 
        temp = new_affine[1,1:3] 
        new_affine[1,1:3] = temp[::-1]
        
        # IS (z) is swapped with PA (y)
        new_affine[2]=affine[1]
        temp = new_affine[2,1:3] 
        new_affine[2,1:3] = temp[::-1]

        # Put non diagonal elements to zero
        A = new_affine[:3, :3]
        off_diag_mask = ~np.eye(3, dtype=bool)
        if np.any(np.abs(A[off_diag_mask]) > 1e-8):
            print("Warning: Non-diagonal elements detected in affine. "
                  "Forcing diagonal orientation (removing shear/rotation).")
            
            # zero off-diagonal elements
            new_affine[:3, :3] = np.diag(np.diag(A))
        
        # Save the new image
        img = nib.load(file_path)
        new_img = nib.Nifti1Image(img.get_fdata(), affine=new_affine, header=img.header)
        nib.save(new_img, file_path)
        
        exe = os.path.join(cfg["fsl_path"], "fslreorient2std")
        call = [exe,
            f'{file_path} {file_path}']
        os.system(' '.join(call))
    
        
    else:
        print('Careful, your asked swap direction is not programmed. Please edit the script to keep the header information and confirm that the left right directions are correct')



def union_niftis(file_paths, output_path):

    # Load the NIfTI files
    images = [nib.load(path) for path in file_paths]

    # Get the data from the NIfTI files
    image_data = [img.get_fdata() for img in images]

    # Check that all images have the same shape
    shape = image_data[0].shape
    if not all(img.shape == shape for img in image_data):
        raise ValueError("All input images must have the same dimensions")

    # Perform the union operation (logical OR) on the images
    union_data = np.zeros(shape)
    for data in image_data:
        union_data = np.logical_or(union_data, data)

    # Convert the boolean mask back to the original data type
    union_data = union_data.astype(image_data[0].dtype)

    # Create a new NIfTI image with the union data
    union_img = nib.Nifti1Image(union_data, images[0].affine, images[0].header)

    # Save the resulting image
    nib.save(union_img, output_path)


def apply_mrconvert(input_path, output_path, axis_no, axis_low, axis_high, cfg):

    exe = os.path.join(cfg["mrtrix_path"], "mrconvert")
    call = [exe,
            f'{input_path}',
            f'{output_path}',
            f'-coord {str(axis_no) + " " + str(axis_low) + ":" + str(axis_high)}',
            f'-force']

    os.system(' '.join(call))


def convert_to_scans(input_path, init_paths, ext):

    input_img = nib.load(input_path).get_fdata()
    output_paths = replace_string(init_paths, '.nii.gz', ext + '.nii.gz')

    for jj in range(len(output_paths)):

        ref_nii = nib.load(init_paths[jj])
        ref_shape = ref_nii.shape
        no_vols = ref_shape[-1]

        temp_nifti = input_img[:, :, :, :no_vols]
        input_img = np.delete(input_img, np.s_[:no_vols], axis=-1)

        array_to_nii(ref_nii, temp_nifti, output_paths[jj])

    if input_img.shape[-1] != 0:
        raise ValueError("There is remaining data in the initial nifti!")

def concat_niftis(list_niftis, output_path, opt):
    ''' Concatenates volumes from a list of NIfTI files.
        If 'all' is selected, all volumes are concatenated.
        If an integer is provided, it selects up to that number of volumes from each NIfTI. '''

    nifti_objs = [nib.load(n) for n in list_niftis]
    volumes = []

    for nifti in nifti_objs:
        img_data = nifti.get_fdata()
        
        # Ensure all images are at least 4D (expand if necessary)
        if img_data.ndim == 3:
            img_data = np.expand_dims(img_data, axis=-1)

        if opt == 'all':
            volumes.append(img_data)
        elif isinstance(opt, int):
            volumes.append(img_data[:, :, :, :opt])
        else:
            raise ValueError("Please enter either 'all' or an integer number of volumes!")

    combined_nifti = np.concatenate(volumes, axis=3)
    array_to_nii(nifti_objs[0], combined_nifti, output_path)

def unconcat_niftis(nifti_path, delta_path, cfg):
    from collections import defaultdict

    # Load the 4D NIfTI file
    img = nib.load(nifti_path)
    data = img.get_fdata()
    affine = img.affine
    header = img.header

    # Load delta values
    deltas = [int(x) for x in read_numeric_txt(delta_path)[0]]
     
    if len(deltas) != data.shape[3]:
        raise ValueError("Number of delta values does not match number of 3D volumes in NIfTI file.")

    # Group indices by delta value
    delta_groups = defaultdict(list)
    for idx, delta in enumerate(deltas):
        delta_groups[int(float(delta))].append(idx)

    # Write separate NIfTI files for each delta group
    for delta, indices in delta_groups.items():
        sub_data = data[..., indices]
        new_img = nib.Nifti1Image(sub_data, affine, header)
        
        # Make folder per delta
        delta_folder = os.path.join(os.path.join(os.path.dirname(nifti_path)), f'Delta_{delta}')
        os.makedirs(delta_folder, exist_ok=True)
        
        name = os.path.basename(nifti_path).replace('allDelta-allb', f'Delta_{delta}')
        output_path = os.path.join(delta_folder, f'{name}')
        nib.save(new_img, output_path)
        print(f"Saved: {output_path}")
        
        bvecs_path = re.sub(r'(Delta_\d+_).*$', r'\1bvecsRotated.txt', output_path)
        bvals_path = re.sub(r'(Delta_\d+_).*$', r'\1bvalsNom.txt', output_path)

        nifti_to_mif(output_path, bvecs_path, bvals_path, output_path.replace('.nii.gz','.mif'), cfg)
        print(f"Converted to mif: {output_path}")
        
        mask_path         = nifti_path.replace('dwi_dn_gc_ec.nii.gz','mask.nii.gz')
        new_mask_path     = os.path.join(delta_folder,'mask.nii.gz')
        mask_dil_path     = nifti_path.replace('dwi_dn_gc_ec.nii.gz','mask_dil.nii.gz')
        new_mask_dil_path = os.path.join(delta_folder,'mask_dil.nii.gz')
        copy_files([mask_path,mask_dil_path],[new_mask_path,new_mask_dil_path])
        print(f"Copyied mask files: {output_path}")

        sigma_path         = nifti_path.replace('dwi_dn_gc_ec.nii.gz','dwi_dn_sigma.nii.gz')
        new_sigma_path     = os.path.join(delta_folder,'dwi_dn_sigma.nii.gz')
        copy_files([sigma_path],[new_sigma_path])
        print(f"Copyied sigma files: {output_path}")

        

def array_to_nii(in_img, in_array, out_img):

    if isinstance(in_img, nib.nifti1.Nifti1Image):
        img = in_img
    else:
        img = nib.load(in_img)

    aff, hdr = img.affine, img.header
    newimg = nib.Nifti1Image(in_array, affine=aff, header=hdr)
    nib.save(newimg, out_img)


def raw_to_nifti(input_path, output_path, cfg, scans):

    if not os.listdir(output_path):

        # Convert data
        # call = [os.environ['HOME']+f'/anaconda3/envs/Dicomifier/bin/dicomifier ',
        #         f'to-nifti',
        #         f'-z',
        #         f'{input_path}',
        #         f'{output_path}']
        # os.system(' '.join(call))
        sources = [os.path.join(input_path, str(s)) for s in scans]

        cmd = [cfg["conda_exe"], "run", "-n", "Dicomifier",
            "dicomifier",
            "to-nifti",
            "-z",
            *sources,
            output_path,
        ]
        subprocess.run(cmd, check=True)

        # Retrieve directory in which the converted data is
        data_dir_lvl1 = get_subdir(output_path)
        data_dir_lvl2 = get_subdir(data_dir_lvl1)

        # Move all files and subdirectories to the main scan path
        destination_path = os.path.abspath(
            os.path.join(data_dir_lvl2, '..', '..'))
        for item in os.listdir(data_dir_lvl2):
            item_path = os.path.join(data_dir_lvl2, item)
            destination_item_path = os.path.join(destination_path, item)
            shutil.move(item_path, destination_item_path)
        # Remove the empty source_path
        os.rmdir(data_dir_lvl2)
        os.rmdir(data_dir_lvl1)

    else:
        print('Data was already converted!')

##### REGISTRATION #####

def antsreg_full(fixed_path, moving_path, out_transform, cfg, lesion_mask_moving=None):
 
    out_im = out_transform + '.nii.gz'
    
    exe = os.path.join(cfg["ants_path"], "antsRegistration")
    call = [f'{exe} -d 3 --interpolation Linear',
            f'--winsorize-image-intensities [0.01,0.99] --use-histogram-matching 1 ',
            f'--initial-moving-transform [{fixed_path}, {moving_path},1]',
            f'--transform Rigid[0.1] --convergence [1000x500x250x0,1e-6,10] --shrink-factors 12x8x4x1 --smoothing-sigmas 5x4x3x1vox ',
            f'--metric MI[{fixed_path}, {moving_path},1,32,Regular,0.25]',
            #f'--metric CC[{fixed_path}, {moving_path},0.5,4]',
            f'--transform Affine[0.15] --convergence [1000x500x250x0,1e-6,10] --shrink-factors 12x8x4x1 --smoothing-sigmas 5x4x3x1vox ',
            f'--metric MI[{fixed_path}, {moving_path},1,32,Regular,0.25]', \
            #f'--metric CC[{fixed_path}, {moving_path},0.5,4]' ,\
            f'--transform SyN[0.05,3,0] --convergence [200x100x50x20,1e-6,10] --shrink-factors 8x4x2x1 --smoothing-sigmas 3x2x1x0vox ', \
            f'--metric MI[{fixed_path}, {moving_path},0.5,32,Random,0.25]' ,\
            f'--metric CC[{fixed_path}, {moving_path},0.5,4]']
           # f'-o [{out_transform},{out_im}] ']
        
    if lesion_mask_moving:
        call += ['-x', f'[,{lesion_mask_moving}]']

    call.append(f'-o [{out_transform},{out_im}] ')

    print(' '.join(call))
    os.system(' '.join(call))

# def antsreg(fixed_path, moving_path, out_transform):

#     out_im = out_transform + '.nii.gz'
    
#     call = [f'antsRegistration -d 3 --interpolation Linear',
#             f'--winsorize-image-intensities [0.005,0.995] --use-histogram-matching 0 ',
#             f'--initial-moving-transform [{fixed_path}, {moving_path},1]',
#             f'--transform Rigid[0.1] --convergence [1000x500x250x0,1e-7,10] --shrink-factors 12x8x4x1 --smoothing-sigmas 5x4x3x1vox ',
#             f'--metric MI[{fixed_path}, {moving_path},0.5,32,Regular,0.25]',
#             #f'--metric CC[{fixed_path}, {moving_path},0.5,4]',
#             f'--transform Affine[0.15] --convergence [1000x500x250x0,1e-7,10] --shrink-factors 12x8x4x1 --smoothing-sigmas 5x4x3x1vox ',
#             f'--metric MI[{fixed_path}, {moving_path},1.25,32,Random,0.25]', \
#             f'--metric CC[{fixed_path}, {moving_path},0.5,4]' ,\
#             f'--transform SyN[0.1,4,0] --convergence [100x70x50x20,1e-7,10] --shrink-factors 8x4x2x1 --smoothing-sigmas 3x2x1x0vox ', \
#             f'--metric MI[{fixed_path}, {moving_path},1.25,32,Random,0.25]' ,\
#             f'--metric CC[{fixed_path}, {moving_path},1,4]', \
#             f'-o [{out_transform},{out_im}] ']

#     print(' '.join(call))
#     os.system(' '.join(call))

# def antsreg_simple2(fixed_path, moving_path, out_transform, cfg, lesion_mask_moving=None):

#     out_im = out_transform + '.nii.gz'
    
#     exe = os.path.join(cfg["ants_path"], "antsRegistration")
#     call = [f'{exe} -d 3 --interpolation Linear',
#             f'--winsorize-image-intensities [0.005,0.995] --use-histogram-matching 1 ',
#             f'--initial-moving-transform [{fixed_path}, {moving_path},1]',
#             f'--transform Rigid[0.1] --convergence [1000x500x250x0,1e-6,10] --shrink-factors 12x8x4x1 --smoothing-sigmas 5x4x3x1vox',
#             f'--metric MI[{fixed_path}, {moving_path},1,32,Regular,0.25]',
#             #f'--metric CC[{fixed_path}, {moving_path},0.5,4]',
#             #f'--transform Affine[0.15] --convergence [1000x500x250x0,1e-6,10] --shrink-factors 12x8x4x1 --smoothing-sigmas 5x4x3x1vox ',
#             #f'--metric MI[{fixed_path}, {moving_path},1,32,Regular,0.25]', \
#             # f'--metric CC[{fixed_path}, {moving_path},0.5,4]' ,\
#             #f'--transform SyN[0.1,4,0] --convergence [100x70x50x20,1e-7,10] --shrink-factors 8x4x2x1 --smoothing-sigmas 3x2x1x0vox ', \
#             #f'--metric MI[{fixed_path}, {moving_path},1.25,32,Random,0.25]' ,\
#             #f'--metric CC[{fixed_path}, {moving_path},1,4]', \
#             ]
        
#     if lesion_mask_moving:
#         call += ['-x', f'[,{lesion_mask_moving}]']
  
#     call.append(f'-o [{out_transform},{out_im}] ')
  
#     print(' '.join(call))
#     os.system(' '.join(call))
    
def antsreg_simple(fixed_path, moving_path, out_transform, cfg, lesion_mask_moving=None):

    out_im = out_transform + '.nii.gz'
    
    exe = os.path.join(cfg["ants_path"], "antsRegistration")
    call = [f'{exe} -d 3 --interpolation Linear',
            f'--winsorize-image-intensities [0.005,0.995] --use-histogram-matching 1 ',
            f'--initial-moving-transform [{fixed_path}, {moving_path},1]',
            f'--transform Rigid[0.1] --convergence [1000x500x250x0,1e-6,10] --shrink-factors 12x8x4x1 --smoothing-sigmas 5x4x3x1vox ',
            f'--metric MI[{fixed_path}, {moving_path},1,32,Regular,0.25]',
            #f'--metric CC[{fixed_path}, {moving_path},0.5,4]',
            #f'--transform Affine[0.15] --convergence [1000x500x250x0,1e-6,10] --shrink-factors 12x8x4x1 --smoothing-sigmas 5x4x3x1vox ',
            #f'--metric MI[{fixed_path}, {moving_path},1,32,Regular,0.25]', \
            # f'--metric CC[{fixed_path}, {moving_path},0.5,4]' ,\
            #f'--transform SyN[0.1,4,0] --convergence [100x70x50x20,1e-7,10] --shrink-factors 8x4x2x1 --smoothing-sigmas 3x2x1x0vox ', \
            #f'--metric MI[{fixed_path}, {moving_path},1.25,32,Random,0.25]' ,\
            #f'--metric CC[{fixed_path}, {moving_path},1,4]', \
            ]
        
    if lesion_mask_moving:
        call += ['-x', f'[,{lesion_mask_moving}]']
  
    call.append(f'-o [{out_transform},{out_im}] ')
  
    print(' '.join(call))
    os.system(' '.join(call))
      
def antsreg_Affine(fixed_path, moving_path, out_transform, cfg):

    out_im = out_transform + '.nii.gz'
    
    exe = os.path.join(cfg["ants_path"], "antsRegistration")
    call = [f'{exe} -d 3 --interpolation Linear',
            f'--winsorize-image-intensities [0.005,0.995] --use-histogram-matching 1 ',
            f'--initial-moving-transform [{fixed_path}, {moving_path},1]',
            f'--transform Rigid[0.1] --convergence [1000x500x250x0,1e-6,10] --shrink-factors 12x8x4x1 --smoothing-sigmas 5x4x3x1vox ',
            f'--metric MI[{fixed_path}, {moving_path},1,32,Regular,0.25]',
            #f'--metric CC[{fixed_path}, {moving_path},0.5,4]',
            f'--transform Affine[0.15] --convergence [1000x500x250x0,1e-6,10] --shrink-factors 12x8x4x1 --smoothing-sigmas 5x4x3x1vox ',
            f'--metric MI[{fixed_path}, {moving_path},1,32,Regular,0.25]', \
            # f'--metric CC[{fixed_path}, {moving_path},0.5,4]' ,\
            #f'--transform SyN[0.1,4,0] --convergence [100x70x50x20,1e-7,10] --shrink-factors 8x4x2x1 --smoothing-sigmas 3x2x1x0vox ', \
            #f'--metric MI[{fixed_path}, {moving_path},1.25,32,Random,0.25]' ,\
            #f'--metric CC[{fixed_path}, {moving_path},1,4]', \
            f'-o [{out_transform},{out_im}] ']

    print(' '.join(call))
    os.system(' '.join(call))
    
def antsreg_atlas(fixed_path, moving_path, out_transform, cfg):

    out_im = out_transform + '.nii.gz'

    #moving_path = new_path
    exe = os.path.join(cfg["ants_path"], "antsRegistration")
    call = [f'{exe} -d 3 --interpolation Linear',
            f'--winsorize-image-intensities [0.005,0.995] --use-histogram-matching 0 ',
            f'--initial-moving-transform [{fixed_path}, {moving_path},1]',
            f'--transform Rigid[0.1] --convergence [1000x500x250x0,1e-7,10] --shrink-factors 12x8x4x1 --smoothing-sigmas 5x4x3x1vox ',
            f'--metric MI[{fixed_path}, {moving_path},0.5,32,Regular,0.25]',
            #f'--metric CC[{fixed_path}, {moving_path},0.5,4]',
            f'--transform Affine[0.15] --convergence [1000x500x250x0,1e-7,10] --shrink-factors 12x8x4x1 --smoothing-sigmas 5x4x3x1vox ',
            f'--metric MI[{fixed_path}, {moving_path},1.25,32,Random,0.25]', \
            # f'--metric CC[{fixed_path}, {moving_path},0.5,4]' ,\
            f'--transform SyN[0.1,4,0] --convergence [100x70x50x20,1e-7,10] --shrink-factors 8x4x2x1 --smoothing-sigmas 3x2x1x0vox ', \
            f'--metric MI[{fixed_path}, {moving_path},1.25,32,Random,0.25]' ,\
            f'--metric CC[{fixed_path}, {moving_path},0.5,4]', \
            f'-o [{out_transform},{out_im}] --verbose ']

    print(' '.join(call))
    os.system(' '.join(call))
    
def antsreg_syn(fixed_path, moving_path, output_prefix, transformation, cfg):

    exe = os.path.join(cfg["ants_path"], "antsRegistrationSyN.sh")
    call = [f'{exe} -d 3',
            f'-f {fixed_path}',
            f'-m {moving_path}',
            f'-o {output_prefix}',
            f'-t {transformation}']

    print(' '.join(call))
    os.system(' '.join(call))


        

def ants_apply_transforms_simple(input_path, ref_path, output_path, transf_1, cfg, extra=None):
    
    for ii in range(len(input_path)):
        input_temp = input_path[ii]
        output_temp = output_path[ii]

        exe = os.path.join(cfg["ants_path"], "antsApplyTransforms")
        call = [exe,
            '-d 3',
            f'-i {input_temp}', \
            f'-r {ref_path}', \
            f'-t {transf_1}', \
            f'-o {output_temp}']

        # Add more arguments if exists
        if extra is not None:
            if isinstance(extra, str):
               call.extend(extra.split())  # split the string into a list
            else:
               call.extend(extra)


        # Always verbose
        call.append('--verbose')

        # Run the command
        print(' '.join(call))
        os.system(' '.join(call))
        
def ants_apply_transforms_keep_input_grid(input_path, output_path, transf_1, cfg, extra=None):

    for ii in range(len(input_path)):
        input_temp = input_path[ii]
        output_temp = output_path[ii]

        exe = os.path.join(cfg["ants_path"], "antsApplyTransforms")
        call = [exe,
            '-d 3',
            f'-i {input_temp}', \
            f'-r {input_temp}', \
            f'-t {transf_1}', \
            f'-o {output_temp}',]

        # Add more arguments if exists
        if extra is not None:
            if isinstance(extra, str):
               call.extend(extra.split())  # split the string into a list
            else:
               call.extend(extra)


        # Always verbose
        call.append('--verbose')

        # Run the command
        print(' '.join(call))
        os.system(' '.join(call))
        
def ants_apply_transforms_simple_4D(input_path, ref_path, output_path, transf_1, cfg, extra=None):
    
    for ii in range(len(input_path)):
        input_temp = input_path[ii]
        output_temp = output_path[ii]
    
        img_4d = nib.load(input_temp)
        data_4d = img_4d.get_fdata()
        orig_affine = img_4d.affine
        orig_header = img_4d.header.copy()
    

        # Ensure it's 4D
        if data_4d.ndim != 4:
            raise ValueError(f"Expected 4D image, got shape {data_4d.shape}")
    
        transformed_volumes = []
        for vol_idx in range(data_4d.shape[3]):
            vol_data = data_4d[..., vol_idx]
    
            with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as temp_infile:
                temp_in = temp_infile.name
                nib.Nifti1Image(vol_data, orig_affine, orig_header).to_filename(temp_in)
    
            with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as temp_outfile:
                temp_out = temp_outfile.name
    
            dim = 3  # Since each volume is 3D
            exe = os.path.join(cfg["ants_path"], "antsApplyTransforms")
            call = [exe,
                f'-d {dim}',
                f'-i {temp_in}',
                f'-r {ref_path}',
                f'-t {transf_1}',
                f'-o {temp_out}'
            ]
    
            if extra is not None:
                if isinstance(extra, str):
                    call.extend(extra.split())
                else:
                    call.extend(extra)
    
            call.append('--verbose')
            print(' '.join(call))
            os.system(' '.join(call))
    
            # Load transformed volume
            transformed_vol = nib.load(temp_out).get_fdata().astype(np.float32)
            transformed_volumes.append(transformed_vol)
            affine = nib.load(temp_out).affine
            header = nib.load(temp_out).header
            
            # Cleanup temp files
            os.remove(temp_in)
            os.remove(temp_out)
    
        # Stack back into 4D
        transformed_4d = np.stack(transformed_volumes, axis=3)
        out_img = nib.Nifti1Image(transformed_4d, affine, header)
        out_img.to_filename(output_temp)
   
    
    

def ants_apply_transforms(input_path, ref_path, output_path, transf_1, transf_2, cfg, extra=None):  # input_type

    for ii in range(len(input_path)):

        input_temp = input_path[ii]
        output_temp = output_path[ii]

        exe = os.path.join(cfg["ants_path"], "antsApplyTransforms")
        call = [exe,
                f'-d 3', \
                # f'-e {input_type}', \
                f'-i {input_temp}', \
                f'-r {ref_path}', \
                f'-t {transf_1}', \
                f'-t {transf_2}', \
                f'-o {output_temp}']
        
        # Add more arguments if exists
        if extra is not None:
            if isinstance(extra, str):
               call.extend(extra.split())  # split the string into a list
            else:
               call.extend(extra)


        # Always verbose
        call.append('--verbose')

        # Run the command
        print(' '.join(call))
        os.system(' '.join(call))



##### MRS voxel #####

def read_header_file_info(file_path, keys_single, keys_array):
    """Read information from the method file

    :param file_path: path to the header file
    :type file_path: str or pathlib.Path
    :param keys_single: List of header keys that are a single value
    :type keys_single: list of str
    :param keys_array: List of header keys that have array values
    :type keys_array: list of str
    :return: Dict containing the information
    :rtype: dict
    """
    re_searches = [re.compile(rf'{re.escape(x)}\s*=\s*([^\r\n]+)') for x in keys_single]
    re_searches2 = [re.compile(fr'({x})\=\(\s*(\d+(?:\s*,\s*\d+)*)\s*\)') for x in keys_array]

    with open(file_path) as fp:
        methodlines = fp.readlines()

    method_values = {}
    for line in methodlines:
        for re_ptrn in re_searches:
            match = re.search(re_ptrn, line)
            if match:
                method_values[re_ptrn.pattern.split(r'\s*=\s*')[0]] = float(match.group(1))

    # For array values that occur on the line after
    for idx, line in enumerate(methodlines):
        for re_ptrn in re_searches2:
            match = re.search(re_ptrn, line)
            if match:
                method_values[match[1]] = np.array(
                    methodlines[idx+1].split(' ')).astype('float')

    return method_values

def create_mrs_vx(cfg, method_path, vx_path):
    """
    Compute voxel geometry in the current env, but *write* the voxel NIfTI using antspyx
    inside the 'ants' conda env.
    """

    # 1) Compute voxel info here (pipeline env)
    method_voxel_info = read_header_file_info(
        method_path, [], ["PVM_VoxArrSize", "PVM_VoxArrPosition", "PVM_VoxArrGradOrient"]
    )

    # Make it JSON-serializable (lists, not numpy types)
    payload = {
        "size":   [float(x) for x in method_voxel_info["PVM_VoxArrSize"]],
        "pos":    [float(x) for x in method_voxel_info["PVM_VoxArrPosition"]],
        "orient": [float(x) for x in np.array(method_voxel_info["PVM_VoxArrGradOrient"]).reshape(-1)],
        "out":    str(vx_path),
    }

    # 2) Run antspyx in the 'ants' env, receiving payload from stdin
    with tempfile.TemporaryDirectory() as td:
        payload_path = os.path.join(td, "mrs_voxel_payload.json")
        with open(payload_path, "w") as f:
            json.dump(payload, f)

        # 3) Run antspyx in the 'ants' env reading the JSON file
        code = r"""
import sys, json, numpy as np
import ants  # must be antspyx in this env

payload_path = sys.argv[1]
with open(payload_path, "r") as f:
    p = json.load(f)

size   = p["size"]
pos    = p["pos"]
orient = np.array(p["orient"], dtype=float).reshape(3,3)

img = ants.from_numpy(
    np.ones((1,1,1), dtype=np.float32),
    origin=[-pos[0], -pos[1], pos[2]],
    spacing=size,
    direction=orient
)

ants.image_write(img, p["out"])
"""

        subprocess.run(
            [cfg["conda_exe"], "run", "-n", "ants", "python", "-c", code, payload_path],
            check=True,
        )
    
# def svs_mrs_voxel_from_method_file(method_path,svs_mrs_voxel_path,return_ants_img=True):
#     import ants

   
#     mrs_voxel = ants.from_numpy(np.ones([1,1,1]),
#                 origin=[-method_voxel_info['PVM_VoxArrPosition'][0],-method_voxel_info['PVM_VoxArrPosition'][1],method_voxel_info['PVM_VoxArrPosition'][2]],
#                 spacing=[method_voxel_info['PVM_VoxArrSize'][0],method_voxel_info['PVM_VoxArrSize'][1],method_voxel_info['PVM_VoxArrSize'][2]],
#                 direction=method_voxel_info['PVM_VoxArrGradOrient'].reshape(3,3))
#     ants.image_write(mrs_voxel,svs_mrs_voxel_path)
#     if return_ants_img:
#         return mrs_voxel
#     return 0

# def create_mrs_vx(cfg,method_path,vx_path):
#     # runs on environment ants

#      method_voxel_info = read_header_file_info(method_path, [], ['PVM_VoxArrSize',
#                                                               'PVM_VoxArrPosition',
#                                                               'PVM_VoxArrGradOrient'])
 
#     subprocess.run([
#         cfg["conda_exe"], "run", "-n", "ants", "python", "-c",
#         (
#             "import sys, numpy as np, re;"
#             "sys.path.append(r'{}');"
#             "from custom_functions import svs_mrs_voxel_from_method_file;"
#             "svs_mrs_voxel_from_method_file(r'{}', r'{}')"
#         ).format(
#             os.path.join(cfg['code_path']),
#             method_path,                 
#             vx_path        
#         )
#     ], check=True)
            
def resample_mrs_voxel(vx_path, anat_orig_path, vx_path_resampled,cfg):

    exe = os.path.join(cfg["ants_path"], "antsApplyTransforms")
    cmd = [exe,
        "-d", "3",
        "-i", f"{vx_path}",
        "-r", f"{anat_orig_path}",
        "-o", f"{vx_path_resampled}",
        "-n", "NearestNeighbor",
        "-t", "identity"
    ]
    
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    
    
##### ATLAS HANDLE #####

def split_mask_left_right(mask, bids_strc_reg, vx_middle, ROI):
    
    temp = nib.load(bids_strc_reg.get_path(f'mask_{ROI}.nii.gz'))
    mask_left = mask.copy()
    mask_left[0:vx_middle+1,:,:]=0
    mask_right = mask.copy()
    mask_right[vx_middle:-1:,:]=0
    masked_img = nib.Nifti1Image(mask_left, affine=temp.affine, header=temp.header)
    nib.save(masked_img, bids_strc_reg.get_path(f'mask_{ROI}_left.nii.gz'))
    masked_img = nib.Nifti1Image(mask_right, affine=temp.affine, header=temp.header)
    nib.save(masked_img, bids_strc_reg.get_path(f'mask_{ROI}_right.nii.gz'))
    
    return  mask_left, mask_right
      
def create_ROI_mask_fromindx(atlas, atlas_labels, indx, bids_strc_reg):
    
    # Load the atlas data
    template = nib.load(atlas)
    atlas_data = template.get_fdata()
    
    mask_indexes = np.isin(atlas_data, indx)
    masked_data = (mask_indexes > 0).astype(np.uint8)
    

    return masked_data

def get_values_within_ROI(ROI_list, atlas, atlas_labels, TPMs, cfg_tpm_thr, 
                          vx_middle, patterns, maximums, bids_strc_reg, bids_mrs, bids_manual, model_path):
 
     Data      = np.zeros((len(ROI_list), len(patterns)))
     Data_all  = np.empty((len(ROI_list), len(patterns)), dtype=object)
     Data_r    = np.empty((len(ROI_list), len(patterns)), dtype=object)
     Data_l    = np.empty((len(ROI_list), len(patterns)), dtype=object)
     
     for i, ROI in enumerate(ROI_list):
         print(f' Getting model estimates from {ROI}...')

         if ROI == 'voxel_mrs':
             mask = nib.load(bids_mrs.get_path('voxel_mrs.nii.gz')).get_fdata()
             copy_files([bids_mrs.get_path('voxel_mrs.nii.gz')],[bids_strc_reg.get_path(f'mask_voxel_mrs.nii.gz')])
         elif ROI =='voxel_mrs_GM':
             mask = nib.load(bids_mrs.get_path('voxel_mrs.nii.gz')).get_fdata()
             copy_files([bids_mrs.get_path('voxel_mrs.nii.gz')],[bids_strc_reg.get_path(f'mask_voxel_mrs_GM.nii.gz')])
             img = nib.load(bids_strc_reg.get_path(f'mask_voxel_mrs_GM.nii.gz'))
             # GM
             tpm_GM = [f for f in TPMs if 'GM' in f and f and os.path.exists(f)]
             tmp_GM = nib.load(tpm_GM[0]).get_fdata() > cfg_tpm_thr if tpm_GM else np.ones(nib.load(atlas).shape, dtype=bool)
             mask = img.get_fdata()*tmp_GM
             masked_img = nib.Nifti1Image(mask, affine=img.affine, header=img.header)
             nib.save(masked_img, bids_strc_reg.get_path(f'mask_voxel_mrs_GM.nii.gz'))
         elif os.path.exists(bids_manual.get_path(f'mask_{ROI}.nii.gz')):
             mask = nib.load(bids_manual.get_path(f'mask_{ROI}.nii.gz')).get_fdata()
         else:
             mask = create_ROI_mask(atlas, atlas_labels, TPMs, ROI, cfg_tpm_thr, bids_strc_reg)

         # Create left and right versions of the mask
         if os.path.exists(bids_manual.get_path(f'mask_{ROI}.nii.gz')):
             mask_left, mask_right = split_mask_left_right(mask, bids_manual, vx_middle, ROI)
         else:
             mask_left, mask_right = split_mask_left_right(mask, bids_strc_reg, vx_middle, ROI)
         
         # Get value of parameter map
         for j, (pattern, maximum) in enumerate(zip(patterns, maximums)): 
             files = glob.glob(os.path.join(model_path, pattern))
             # Remove files ending with "initialization"
             matched_file = [f for f in files if "initialization" not in os.path.basename(f)]
            # matched_file = glob.glob(os.path.join(model_path, pattern))
             print(f'  Getting model estimates from {pattern} from image {matched_file}...')

             # Filter out files where 'fs' appears in the filename for sandi when looking for f
             if pattern=='*sandi*f*':
                 matched_file = [
                     f for f in matched_file
                     if 'fs' not in os.path.basename(f).lower()  
                 ]
             
             param_img = nib.load(matched_file[0]).get_fdata()
             masked    = param_img[mask > 0]  # Select only voxels inside the ROI
             masked_l  = param_img[mask_left > 0]  # Select only voxels inside the ROI
             masked_r  = param_img[mask_right > 0]  # Select only voxels inside the ROI

             # Remove Nans and voxels that hit the limit threshold as they are not reliable
             masked_clean = masked[~np.isnan(masked)] #& (masked > maximum[0]) & (masked < maximum[1])]
             masked_clean_l = masked_l[~np.isnan(masked_l)] #& (masked_l > maximum[0]) & (masked_l < maximum[1])]
             masked_clean_r = masked_r[~np.isnan(masked_r)] #& (masked_r > maximum[0]) & (masked_r < maximum[1])]
             # masked_clean = masked[~np.isnan(masked)]
             # masked_clean_l = masked_l[~np.isnan(masked_l)]
             # masked_clean_r = masked_r[~np.isnan(masked_r)]
                       
             # Save in matrix
             Data[i, j]     = np.nanmedian(masked_clean) if len(masked_clean) > 0 else np.nan
             Data_all[i, j] = masked_clean if len(masked_clean) > 0 else np.nan
             Data_l[i, j]   = masked_clean_l if len(masked_clean_l) > 0 else np.nan
             Data_r[i, j]   = masked_clean_r if len(masked_clean_r) > 0 else np.nan


     return  Data, Data_all, Data_l, Data_r
