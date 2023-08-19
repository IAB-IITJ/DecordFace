'''
This file provides all the functions required for computing the corruption evaluation metrics:
- mVCE
- mCEI

Refer to the functions and the associated function description for the details about the arguments and the function outputs
'''

# ---------------------------------------------- import necessary libraries

# matrix manipulation
import numpy as np

# ---------------------------------------------- mVCE metric

'''
Prerequisites: 
To run the mVCE metric calculation, the TPR@FPR at the reporting FPRs needs to be tabulated (or must be stored in lists) first. 
An example for the same is shown below for ArcFace R100 Backbone's performance on clean CALFW (severity 0) and CALFW-facet (severity 1 to 5):

+----------------+-------------------+----------------------------------------------------+
|                |                   | TPR@FPR                                            |
+----------------+-------------------+-------+--------+--------+--------+--------+--------+
| FPR            |                   |  0.01 |   0.01 |   0.01 |   0.01 |   0.01 |   0.01 |
+----------------+-------------------+-------+--------+--------+--------+--------+--------+
| severity_level |                   |     0 |      1 |      2 |      3 |      4 |      5 |
+----------------+-------------------+-------+--------+--------+--------+--------+--------+
| backbone_name  | corruption_name   |       |        |        |        |        |        |
+----------------+-------------------+-------+--------+--------+--------+--------+--------+
| r100           | brightness        | 0.926 |  0.925 | 0.9233 |  0.918 |   0.91 | 0.8967 |
+----------------+-------------------+-------+--------+--------+--------+--------+--------+
| r100           | contrast          | 0.926 | 0.9237 | 0.9233 | 0.9163 | 0.8713 | 0.3283 |
+----------------+-------------------+-------+--------+--------+--------+--------+--------+
| r100           | defocus_blur      | 0.926 |  0.926 |  0.924 | 0.9223 |  0.899 | 0.8193 |
+----------------+-------------------+-------+--------+--------+--------+--------+--------+
| r100           | elastic_transform | 0.926 | 0.9227 | 0.9173 | 0.9137 | 0.9107 |  0.891 |
+----------------+-------------------+-------+--------+--------+--------+--------+--------+
| r100           | gaussian_blur     | 0.926 |  0.926 | 0.9247 |  0.921 | 0.9167 | 0.8617 |
+----------------+-------------------+-------+--------+--------+--------+--------+--------+
| r100           | gaussian_noise    | 0.926 |  0.924 | 0.9233 |  0.921 | 0.9073 | 0.8457 |
+----------------+-------------------+-------+--------+--------+--------+--------+--------+
| r100           | glass_blur        | 0.926 | 0.9267 |  0.926 | 0.9177 |  0.914 | 0.8803 |
+----------------+-------------------+-------+--------+--------+--------+--------+--------+
| r100           | impulse_noise     | 0.926 |  0.922 |   0.92 |  0.919 |  0.907 | 0.8647 |
+----------------+-------------------+-------+--------+--------+--------+--------+--------+
| r100           | jpeg_compression  | 0.926 |  0.926 |  0.926 | 0.9243 | 0.9227 | 0.9173 |
+----------------+-------------------+-------+--------+--------+--------+--------+--------+
| r100           | motion_blur       | 0.926 | 0.9257 | 0.9253 | 0.9197 |  0.895 | 0.8617 |
+----------------+-------------------+-------+--------+--------+--------+--------+--------+
| r100           | pixelate          | 0.926 |  0.927 |  0.926 | 0.9263 | 0.9243 | 0.9253 |
+----------------+-------------------+-------+--------+--------+--------+--------+--------+
| r100           | saturate          | 0.926 | 0.9243 | 0.9233 |  0.925 | 0.9143 |  0.915 |
+----------------+-------------------+-------+--------+--------+--------+--------+--------+
| r100           | shot_noise        | 0.926 | 0.9247 | 0.9247 | 0.9203 | 0.8793 | 0.8007 |
+----------------+-------------------+-------+--------+--------+--------+--------+--------+
| r100           | spatter           | 0.926 |  0.925 | 0.9203 | 0.9087 | 0.8717 | 0.7223 |
+----------------+-------------------+-------+--------+--------+--------+--------+--------+
| r100           | speckle_noise     | 0.926 | 0.9243 | 0.9263 |  0.914 |  0.906 | 0.8707 |
+----------------+-------------------+-------+--------+--------+--------+--------+--------+
| r100           | zoom_blur         | 0.926 | 0.9243 |  0.923 |  0.916 | 0.9067 | 0.8587 |
+----------------+-------------------+-------+--------+--------+--------+--------+--------+

The first 4 rows are just headers and won't be required in the function. We will refer to this dataset while explaining the arguments of `get_mVCE` function.
Let's refer the data above as `df` in the rest of the code.
'''


# Function for calculating mVCE metric
def get_mVCE(result_model_names, result_corr_names, result_corr_0, result_corr_1, result_corr_2, result_corr_3, result_corr_4, result_corr_5, num_corruptions=16, severity='overall'):
    '''
    It returns the model name and the computed mVCE and RmVCE metric.
    :result_model_names: It should contain the name or backbone of the model for which we want to compute the mVCE metric. 
                         It should be a list of the same length as `num_corruption`. 
                         Eg- result_model_names = df.iloc[:, :1].values.tolist()
    :result_corr_names: It is the list of corruption names in the same order as the remaining lists in the arguments are passed.
                         Eg- result_corr_names = df.iloc[:, 1:2].values.tolist()
    :result_corr_0,1,2,3,4,5: It is the list of the TPR values at the severity 0 to 5 respectively.
                         Eg- result_corr_0 = df.iloc[:num_rows, 2:3].values.tolist()
                             result_corr_1 = df.iloc[:num_rows, 3:4].values.tolist()
                             result_corr_2 = df.iloc[:num_rows, 4:5].values.tolist()
                             result_corr_3 = df.iloc[:num_rows, 5:6].values.tolist()
                             result_corr_4 = df.iloc[:num_rows, 6:7].values.tolist()
                             result_corr_5 = df.iloc[:num_rows, 7:8].values.tolist()
    :num_corruptions: The number of corruptions to be used for calculating the metric. Defaults to `16`.
    :severity: The particular variant of mVCE that should be returned. Defaults to `overall`. Valid choices are (`low`, `high`, `overall`)
    '''

    # basic check whether the input lists are of equal length and are divisible by the number of corruptions.
    assert len(result_model_names)%num_corruptions == 0
    assert len(result_corr_names)%num_corruptions == 0
    assert len(result_corr_0)%num_corruptions == 0
    assert len(result_corr_1)%num_corruptions == 0
    assert len(result_corr_2)%num_corruptions == 0
    assert len(result_corr_3)%num_corruptions == 0
    assert len(result_corr_4)%num_corruptions == 0
    assert len(result_corr_5)%num_corruptions == 0
    
    # verification accuracy values are in range 0-1
    error_corr_0 = np.array([round(100-(result_corr_0[i][0]*100), 4) for i in range(len(result_corr_0))])   
    error_corr_1 = np.array([round(100-(result_corr_1[i][0]*100), 4) for i in range(len(result_corr_1))])
    error_corr_2 = np.array([round(100-(result_corr_2[i][0]*100), 4) for i in range(len(result_corr_2))])
    error_corr_3 = np.array([round(100-(result_corr_3[i][0]*100), 4) for i in range(len(result_corr_3))])
    error_corr_4 = np.array([round(100-(result_corr_4[i][0]*100), 4) for i in range(len(result_corr_4))])
    error_corr_5 = np.array([round(100-(result_corr_5[i][0]*100), 4) for i in range(len(result_corr_5))])
    
    # stores the computed mVCE and 
    num_models = len(result_model_names)//num_corruptions
    mVCE = np.zeros(num_models)
    RmVCE = np.zeros(num_models)
    backbone_names = []

    # compute the mVCE and RmVCE metric
    for i in range(num_models):
        start = int(i*num_corruptions)
        end = int((i+1)*num_corruptions)

        backbone_names.append(result_model_names[start][0])
        if severity == 'overall':
            mVCE[i] = np.mean([error_corr_1[start:end], error_corr_2[start:end], error_corr_3[start:end], error_corr_4[start:end], error_corr_5[start:end]])
            RmVCE[i] = np.mean([error_corr_1[start:end]-error_corr_0[start:end], error_corr_2[start:end]-error_corr_0[start:end], error_corr_3[start:end]-error_corr_0[start:end], error_corr_4[start:end]-error_corr_0[start:end], error_corr_5[start:end]-error_corr_0[start:end]])
        elif severity == 'low':
            mVCE[i] = np.mean([error_corr_1[start:end], error_corr_2[start:end], error_corr_3[start:end]])
            RmVCE[i] = np.mean([error_corr_1[start:end]-error_corr_0[start:end], error_corr_2[start:end]-error_corr_0[start:end], error_corr_3[start:end]-error_corr_0[start:end]])
        elif severity == 'high':
            mVCE[i] = np.mean([error_corr_4[start:end], error_corr_5[start:end]])
            RmVCE[i] = np.mean([error_corr_4[start:end]-error_corr_0[start:end], error_corr_5[start:end]-error_corr_0[start:end]])

        mVCE[i] = round(mVCE[i], 2)
        RmVCE[i] = round(RmVCE[i], 2)

    # return the metric
    return backbone_names, mVCE, RmVCE

# ---------------------------------------------- mCEI metric
'''
Prerequisites: 
To run the mCEI metric calculation, the average cosine similarity over the clean and benchmark datasets needs to be tabulated (or must be stored in lists) first. 
An example for the same is shown below for ArcFace R100 Backbone's performance on clean CALFW (severity 0) and CALFW-facet (severity 1 to 5):

+----------------+-------------------+------------------------------------------------+
|                |                   | AvgSimScore                                    |
+----------------+-------------------+---+--------+--------+--------+--------+--------+
| severity_level |                   | 0 |      1 |      2 |      3 |      4 |      5 |
+----------------+-------------------+---+--------+--------+--------+--------+--------+
| backbone_name  | corruption_name   |   |        |        |        |        |        |
+----------------+-------------------+---+--------+--------+--------+--------+--------+
| r100           | brightness        | 1 | 0.9713 | 0.9218 | 0.8574 | 0.7892 | 0.7285 |
+----------------+-------------------+---+--------+--------+--------+--------+--------+
| r100           | contrast          | 1 | 0.9316 | 0.9015 | 0.8418 | 0.6608 | 0.2608 |
+----------------+-------------------+---+--------+--------+--------+--------+--------+
| r100           | defocus_blur      | 1 | 0.9627 | 0.9399 | 0.8563 | 0.7361 | 0.5792 |
+----------------+-------------------+---+--------+--------+--------+--------+--------+
| r100           | elastic_transform | 1 | 0.8877 | 0.8459 | 0.7879 | 0.7421 | 0.6861 |
+----------------+-------------------+---+--------+--------+--------+--------+--------+
| r100           | gaussian_blur     | 1 | 0.9788 | 0.9497 | 0.8974 | 0.8249 | 0.6502 |
+----------------+-------------------+---+--------+--------+--------+--------+--------+
| r100           | gaussian_noise    | 1 | 0.9406 | 0.9008 | 0.8412 | 0.7637 | 0.6314 |
+----------------+-------------------+---+--------+--------+--------+--------+--------+
| r100           | glass_blur        | 1 | 0.9653 | 0.9398 | 0.8483 | 0.7932 | 0.6822 |
+----------------+-------------------+---+--------+--------+--------+--------+--------+
| r100           | impulse_noise     | 1 | 0.9279 | 0.8866 | 0.8524 | 0.7724 | 0.6623 |
+----------------+-------------------+---+--------+--------+--------+--------+--------+
| r100           | jpeg_compression  | 1 | 0.9675 | 0.9512 | 0.9377 | 0.8969 | 0.8382 |
+----------------+-------------------+---+--------+--------+--------+--------+--------+
| r100           | motion_blur       | 1 | 0.9657 | 0.9323 | 0.8568 | 0.7354 | 0.6441 |
+----------------+-------------------+---+--------+--------+--------+--------+--------+
| r100           | pixelate          | 1 | 0.9849 | 0.9907 | 0.9732 | 0.9575 |   0.94 |
+----------------+-------------------+---+--------+--------+--------+--------+--------+
| r100           | saturate          | 1 | 0.9617 | 0.9422 | 0.9417 | 0.8254 | 0.8125 |
+----------------+-------------------+---+--------+--------+--------+--------+--------+
| r100           | shot_noise        | 1 | 0.9282 | 0.8786 | 0.8195 | 0.7107 |  0.605 |
+----------------+-------------------+---+--------+--------+--------+--------+--------+
| r100           | spatter           | 1 | 0.9694 | 0.8687 | 0.7758 | 0.7028 | 0.5377 |
+----------------+-------------------+---+--------+--------+--------+--------+--------+
| r100           | speckle_noise     | 1 | 0.9371 | 0.9109 |  0.831 | 0.7783 | 0.6994 |
+----------------+-------------------+---+--------+--------+--------+--------+--------+
| r100           | zoom_blur         | 1 | 0.9188 | 0.8693 | 0.7772 | 0.7193 | 0.6227 |
+----------------+-------------------+---+--------+--------+--------+--------+--------+

The first 3 rows are just headers and won't be required in the function. We will refer to this dataset while explaining the arguments of `get_mVCE` function.
Let's refer the data above as `dfsim` in the rest of the code.
'''
# # Function for calculating mCEI metric
def get_mCEI(result_model_names, result_corr_names, result_corr_0, result_corr_1, result_corr_2, result_corr_3, result_corr_4, result_corr_5, num_corruptions=16, severity='overall'):
    '''
    It returns the model name and the computed mCEI metric.
    :result_model_names: It should contain the name or backbone of the model for which we want to compute the mVCE metric. 
                         It should be a list of the same length as `num_corruption`. 
                         Eg- result_model_names = dfsim.iloc[:num_rows, :1].values.tolist()
    :result_corr_names: It is the list of corruption names in the same order as the remaining lists in the arguments are passed.
                         Eg- result_corr_names = dfsim.iloc[:num_rows, 1:2].values.tolist()
    :result_corr_0,1,2,3,4,5: It is the list of the TPR values at the severity 0 to 5 respectively.
                         Eg- result_corr_0 = dfsim.iloc[:num_rows, 2:3].values.tolist()
                             result_corr_1 = dfsim.iloc[:num_rows, 3:4].values.tolist()
                             result_corr_2 = dfsim.iloc[:num_rows, 4:5].values.tolist()
                             result_corr_3 = dfsim.iloc[:num_rows, 5:6].values.tolist()
                             result_corr_4 = dfsim.iloc[:num_rows, 6:7].values.tolist()
                             result_corr_5 = dfsim.iloc[:num_rows, 7:8].values.tolist()
    :num_corruptions: The number of corruptions to be used for calculating the metric. Defaults to `16`.
    :severity: The particular variant of mCEI that should be returned. Defaults to `overall`. Valid choices are (`low`, `high`, `overall`)
    '''
    assert len(result_model_names)%num_corruptions == 0
    assert len(result_corr_names)%num_corruptions == 0
    assert len(result_corr_0)%num_corruptions == 0
    assert len(result_corr_1)%num_corruptions == 0
    assert len(result_corr_2)%num_corruptions == 0
    assert len(result_corr_3)%num_corruptions == 0
    assert len(result_corr_4)%num_corruptions == 0
    assert len(result_corr_5)%num_corruptions == 0
    
    num_models = len(result_model_names)//num_corruptions
    mCEI = np.zeros(num_models)
    backbone_names = []

    for i in range(num_models):
        start = int(i*num_corruptions)
        end = int((i+1)*num_corruptions)

        backbone_names.append(result_model_names[start][0])
        if severity == 'overall':
            mCEI[i] = np.mean([result_corr_1[start:end], result_corr_2[start:end], result_corr_3[start:end], result_corr_4[start:end], result_corr_5[start:end]]) * 100
        elif severity == 'low':
            mCEI[i] = np.mean([result_corr_1[start:end], result_corr_2[start:end], result_corr_3[start:end]]) * 100
        elif severity == 'high':
            mCEI[i] = np.mean([result_corr_4[start:end], result_corr_5[start:end]]) * 100

        mCEI[i] = round(mCEI[i], 2)

    return backbone_names, mCEI