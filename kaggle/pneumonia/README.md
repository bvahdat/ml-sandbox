# What is this codebase about?

See [here](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).

# How to run

I use [miniconda](https://docs.conda.io/en/latest/miniconda.html) through [Homebrew](https://formulae.brew.sh/cask/miniconda) on my Mac to setup the required environment:

```
conda create -n pneumonia -c conda-forge matplotlib tensorflow python=3.10 
conda activate pneumonia
python pneumonia.py
```

Which yields an output similar to:

```
Found 5216 files belonging to 2 classes.
Metal device set to: Apple M1 Ultra

systemMemory: 128.00 GB
maxCacheSize: 48.00 GB

2023-03-30 17:35:30.895978: I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:305] Could not identify NUMA node of platform GPU ID 0, defaulting to 0. Your kernel may not have been built with NUMA support.
2023-03-30 17:35:30.896251: I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:271] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 0 MB memory) -> physical PluggableDevice (device: 0, name: METAL, pci bus id: <undefined>)
Found 16 files belonging to 2 classes.
Found 624 files belonging to 2 classes.
in the training set 2 class names to classify the images for: ['NORMAL', 'PNEUMONIA']
2023-03-30 17:35:31.037905: W tensorflow/core/platform/profile_utils/cpu_utils.cc:128] Failed to get CPU frequency: 0 Hz
image batch shape: (32, 224, 224, 3)
image label shape: (32,)
Model: "mobilenetv2_1.00_224"
__________________________________________________________________________________________________
 Layer (type)                   Output Shape         Param #     Connected to                     
==================================================================================================
 input_1 (InputLayer)           [(None, 224, 224, 3  0           []                               
                                )]                                                                
                                                                                                  
 Conv1 (Conv2D)                 (None, 112, 112, 32  864         ['input_1[0][0]']                
                                )                                                                 
                                                                                                  
 bn_Conv1 (BatchNormalization)  (None, 112, 112, 32  128         ['Conv1[0][0]']                  
                                )                                                                 
                                                                                                  
 Conv1_relu (ReLU)              (None, 112, 112, 32  0           ['bn_Conv1[0][0]']               
                                )                                                                 
                                                                                                  
 expanded_conv_depthwise (Depth  (None, 112, 112, 32  288        ['Conv1_relu[0][0]']             
 wiseConv2D)                    )                                                                 
                                                                                                  
 expanded_conv_depthwise_BN (Ba  (None, 112, 112, 32  128        ['expanded_conv_depthwise[0][0]']
 tchNormalization)              )                                                                 
                                                                                                  
 expanded_conv_depthwise_relu (  (None, 112, 112, 32  0          ['expanded_conv_depthwise_BN[0][0
 ReLU)                          )                                ]']                              
                                                                                                  
 expanded_conv_project (Conv2D)  (None, 112, 112, 16  512        ['expanded_conv_depthwise_relu[0]
                                )                                [0]']                            
                                                                                                  
 expanded_conv_project_BN (Batc  (None, 112, 112, 16  64         ['expanded_conv_project[0][0]']  
 hNormalization)                )                                                                 
                                                                                                  
 block_1_expand (Conv2D)        (None, 112, 112, 96  1536        ['expanded_conv_project_BN[0][0]'
                                )                                ]                                
                                                                                                  
 block_1_expand_BN (BatchNormal  (None, 112, 112, 96  384        ['block_1_expand[0][0]']         
 ization)                       )                                                                 
                                                                                                  
 block_1_expand_relu (ReLU)     (None, 112, 112, 96  0           ['block_1_expand_BN[0][0]']      
                                )                                                                 
                                                                                                  
 block_1_pad (ZeroPadding2D)    (None, 113, 113, 96  0           ['block_1_expand_relu[0][0]']    
                                )                                                                 
                                                                                                  
 block_1_depthwise (DepthwiseCo  (None, 56, 56, 96)  864         ['block_1_pad[0][0]']            
 nv2D)                                                                                            
                                                                                                  
 block_1_depthwise_BN (BatchNor  (None, 56, 56, 96)  384         ['block_1_depthwise[0][0]']      
 malization)                                                                                      
                                                                                                  
 block_1_depthwise_relu (ReLU)  (None, 56, 56, 96)   0           ['block_1_depthwise_BN[0][0]']   
                                                                                                  
 block_1_project (Conv2D)       (None, 56, 56, 24)   2304        ['block_1_depthwise_relu[0][0]'] 
                                                                                                  
 block_1_project_BN (BatchNorma  (None, 56, 56, 24)  96          ['block_1_project[0][0]']        
 lization)                                                                                        
                                                                                                  
 block_2_expand (Conv2D)        (None, 56, 56, 144)  3456        ['block_1_project_BN[0][0]']     
                                                                                                  
 block_2_expand_BN (BatchNormal  (None, 56, 56, 144)  576        ['block_2_expand[0][0]']         
 ization)                                                                                         
                                                                                                  
 block_2_expand_relu (ReLU)     (None, 56, 56, 144)  0           ['block_2_expand_BN[0][0]']      
                                                                                                  
 block_2_depthwise (DepthwiseCo  (None, 56, 56, 144)  1296       ['block_2_expand_relu[0][0]']    
 nv2D)                                                                                            
                                                                                                  
 block_2_depthwise_BN (BatchNor  (None, 56, 56, 144)  576        ['block_2_depthwise[0][0]']      
 malization)                                                                                      
                                                                                                  
 block_2_depthwise_relu (ReLU)  (None, 56, 56, 144)  0           ['block_2_depthwise_BN[0][0]']   
                                                                                                  
 block_2_project (Conv2D)       (None, 56, 56, 24)   3456        ['block_2_depthwise_relu[0][0]'] 
                                                                                                  
 block_2_project_BN (BatchNorma  (None, 56, 56, 24)  96          ['block_2_project[0][0]']        
 lization)                                                                                        
                                                                                                  
 block_2_add (Add)              (None, 56, 56, 24)   0           ['block_1_project_BN[0][0]',     
                                                                  'block_2_project_BN[0][0]']     
                                                                                                  
 block_3_expand (Conv2D)        (None, 56, 56, 144)  3456        ['block_2_add[0][0]']            
                                                                                                  
 block_3_expand_BN (BatchNormal  (None, 56, 56, 144)  576        ['block_3_expand[0][0]']         
 ization)                                                                                         
                                                                                                  
 block_3_expand_relu (ReLU)     (None, 56, 56, 144)  0           ['block_3_expand_BN[0][0]']      
                                                                                                  
 block_3_pad (ZeroPadding2D)    (None, 57, 57, 144)  0           ['block_3_expand_relu[0][0]']    
                                                                                                  
 block_3_depthwise (DepthwiseCo  (None, 28, 28, 144)  1296       ['block_3_pad[0][0]']            
 nv2D)                                                                                            
                                                                                                  
 block_3_depthwise_BN (BatchNor  (None, 28, 28, 144)  576        ['block_3_depthwise[0][0]']      
 malization)                                                                                      
                                                                                                  
 block_3_depthwise_relu (ReLU)  (None, 28, 28, 144)  0           ['block_3_depthwise_BN[0][0]']   
                                                                                                  
 block_3_project (Conv2D)       (None, 28, 28, 32)   4608        ['block_3_depthwise_relu[0][0]'] 
                                                                                                  
 block_3_project_BN (BatchNorma  (None, 28, 28, 32)  128         ['block_3_project[0][0]']        
 lization)                                                                                        
                                                                                                  
 block_4_expand (Conv2D)        (None, 28, 28, 192)  6144        ['block_3_project_BN[0][0]']     
                                                                                                  
 block_4_expand_BN (BatchNormal  (None, 28, 28, 192)  768        ['block_4_expand[0][0]']         
 ization)                                                                                         
                                                                                                  
 block_4_expand_relu (ReLU)     (None, 28, 28, 192)  0           ['block_4_expand_BN[0][0]']      
                                                                                                  
 block_4_depthwise (DepthwiseCo  (None, 28, 28, 192)  1728       ['block_4_expand_relu[0][0]']    
 nv2D)                                                                                            
                                                                                                  
 block_4_depthwise_BN (BatchNor  (None, 28, 28, 192)  768        ['block_4_depthwise[0][0]']      
 malization)                                                                                      
                                                                                                  
 block_4_depthwise_relu (ReLU)  (None, 28, 28, 192)  0           ['block_4_depthwise_BN[0][0]']   
                                                                                                  
 block_4_project (Conv2D)       (None, 28, 28, 32)   6144        ['block_4_depthwise_relu[0][0]'] 
                                                                                                  
 block_4_project_BN (BatchNorma  (None, 28, 28, 32)  128         ['block_4_project[0][0]']        
 lization)                                                                                        
                                                                                                  
 block_4_add (Add)              (None, 28, 28, 32)   0           ['block_3_project_BN[0][0]',     
                                                                  'block_4_project_BN[0][0]']     
                                                                                                  
 block_5_expand (Conv2D)        (None, 28, 28, 192)  6144        ['block_4_add[0][0]']            
                                                                                                  
 block_5_expand_BN (BatchNormal  (None, 28, 28, 192)  768        ['block_5_expand[0][0]']         
 ization)                                                                                         
                                                                                                  
 block_5_expand_relu (ReLU)     (None, 28, 28, 192)  0           ['block_5_expand_BN[0][0]']      
                                                                                                  
 block_5_depthwise (DepthwiseCo  (None, 28, 28, 192)  1728       ['block_5_expand_relu[0][0]']    
 nv2D)                                                                                            
                                                                                                  
 block_5_depthwise_BN (BatchNor  (None, 28, 28, 192)  768        ['block_5_depthwise[0][0]']      
 malization)                                                                                      
                                                                                                  
 block_5_depthwise_relu (ReLU)  (None, 28, 28, 192)  0           ['block_5_depthwise_BN[0][0]']   
                                                                                                  
 block_5_project (Conv2D)       (None, 28, 28, 32)   6144        ['block_5_depthwise_relu[0][0]'] 
                                                                                                  
 block_5_project_BN (BatchNorma  (None, 28, 28, 32)  128         ['block_5_project[0][0]']        
 lization)                                                                                        
                                                                                                  
 block_5_add (Add)              (None, 28, 28, 32)   0           ['block_4_add[0][0]',            
                                                                  'block_5_project_BN[0][0]']     
                                                                                                  
 block_6_expand (Conv2D)        (None, 28, 28, 192)  6144        ['block_5_add[0][0]']            
                                                                                                  
 block_6_expand_BN (BatchNormal  (None, 28, 28, 192)  768        ['block_6_expand[0][0]']         
 ization)                                                                                         
                                                                                                  
 block_6_expand_relu (ReLU)     (None, 28, 28, 192)  0           ['block_6_expand_BN[0][0]']      
                                                                                                  
 block_6_pad (ZeroPadding2D)    (None, 29, 29, 192)  0           ['block_6_expand_relu[0][0]']    
                                                                                                  
 block_6_depthwise (DepthwiseCo  (None, 14, 14, 192)  1728       ['block_6_pad[0][0]']            
 nv2D)                                                                                            
                                                                                                  
 block_6_depthwise_BN (BatchNor  (None, 14, 14, 192)  768        ['block_6_depthwise[0][0]']      
 malization)                                                                                      
                                                                                                  
 block_6_depthwise_relu (ReLU)  (None, 14, 14, 192)  0           ['block_6_depthwise_BN[0][0]']   
                                                                                                  
 block_6_project (Conv2D)       (None, 14, 14, 64)   12288       ['block_6_depthwise_relu[0][0]'] 
                                                                                                  
 block_6_project_BN (BatchNorma  (None, 14, 14, 64)  256         ['block_6_project[0][0]']        
 lization)                                                                                        
                                                                                                  
 block_7_expand (Conv2D)        (None, 14, 14, 384)  24576       ['block_6_project_BN[0][0]']     
                                                                                                  
 block_7_expand_BN (BatchNormal  (None, 14, 14, 384)  1536       ['block_7_expand[0][0]']         
 ization)                                                                                         
                                                                                                  
 block_7_expand_relu (ReLU)     (None, 14, 14, 384)  0           ['block_7_expand_BN[0][0]']      
                                                                                                  
 block_7_depthwise (DepthwiseCo  (None, 14, 14, 384)  3456       ['block_7_expand_relu[0][0]']    
 nv2D)                                                                                            
                                                                                                  
 block_7_depthwise_BN (BatchNor  (None, 14, 14, 384)  1536       ['block_7_depthwise[0][0]']      
 malization)                                                                                      
                                                                                                  
 block_7_depthwise_relu (ReLU)  (None, 14, 14, 384)  0           ['block_7_depthwise_BN[0][0]']   
                                                                                                  
 block_7_project (Conv2D)       (None, 14, 14, 64)   24576       ['block_7_depthwise_relu[0][0]'] 
                                                                                                  
 block_7_project_BN (BatchNorma  (None, 14, 14, 64)  256         ['block_7_project[0][0]']        
 lization)                                                                                        
                                                                                                  
 block_7_add (Add)              (None, 14, 14, 64)   0           ['block_6_project_BN[0][0]',     
                                                                  'block_7_project_BN[0][0]']     
                                                                                                  
 block_8_expand (Conv2D)        (None, 14, 14, 384)  24576       ['block_7_add[0][0]']            
                                                                                                  
 block_8_expand_BN (BatchNormal  (None, 14, 14, 384)  1536       ['block_8_expand[0][0]']         
 ization)                                                                                         
                                                                                                  
 block_8_expand_relu (ReLU)     (None, 14, 14, 384)  0           ['block_8_expand_BN[0][0]']      
                                                                                                  
 block_8_depthwise (DepthwiseCo  (None, 14, 14, 384)  3456       ['block_8_expand_relu[0][0]']    
 nv2D)                                                                                            
                                                                                                  
 block_8_depthwise_BN (BatchNor  (None, 14, 14, 384)  1536       ['block_8_depthwise[0][0]']      
 malization)                                                                                      
                                                                                                  
 block_8_depthwise_relu (ReLU)  (None, 14, 14, 384)  0           ['block_8_depthwise_BN[0][0]']   
                                                                                                  
 block_8_project (Conv2D)       (None, 14, 14, 64)   24576       ['block_8_depthwise_relu[0][0]'] 
                                                                                                  
 block_8_project_BN (BatchNorma  (None, 14, 14, 64)  256         ['block_8_project[0][0]']        
 lization)                                                                                        
                                                                                                  
 block_8_add (Add)              (None, 14, 14, 64)   0           ['block_7_add[0][0]',            
                                                                  'block_8_project_BN[0][0]']     
                                                                                                  
 block_9_expand (Conv2D)        (None, 14, 14, 384)  24576       ['block_8_add[0][0]']            
                                                                                                  
 block_9_expand_BN (BatchNormal  (None, 14, 14, 384)  1536       ['block_9_expand[0][0]']         
 ization)                                                                                         
                                                                                                  
 block_9_expand_relu (ReLU)     (None, 14, 14, 384)  0           ['block_9_expand_BN[0][0]']      
                                                                                                  
 block_9_depthwise (DepthwiseCo  (None, 14, 14, 384)  3456       ['block_9_expand_relu[0][0]']    
 nv2D)                                                                                            
                                                                                                  
 block_9_depthwise_BN (BatchNor  (None, 14, 14, 384)  1536       ['block_9_depthwise[0][0]']      
 malization)                                                                                      
                                                                                                  
 block_9_depthwise_relu (ReLU)  (None, 14, 14, 384)  0           ['block_9_depthwise_BN[0][0]']   
                                                                                                  
 block_9_project (Conv2D)       (None, 14, 14, 64)   24576       ['block_9_depthwise_relu[0][0]'] 
                                                                                                  
 block_9_project_BN (BatchNorma  (None, 14, 14, 64)  256         ['block_9_project[0][0]']        
 lization)                                                                                        
                                                                                                  
 block_9_add (Add)              (None, 14, 14, 64)   0           ['block_8_add[0][0]',            
                                                                  'block_9_project_BN[0][0]']     
                                                                                                  
 block_10_expand (Conv2D)       (None, 14, 14, 384)  24576       ['block_9_add[0][0]']            
                                                                                                  
 block_10_expand_BN (BatchNorma  (None, 14, 14, 384)  1536       ['block_10_expand[0][0]']        
 lization)                                                                                        
                                                                                                  
 block_10_expand_relu (ReLU)    (None, 14, 14, 384)  0           ['block_10_expand_BN[0][0]']     
                                                                                                  
 block_10_depthwise (DepthwiseC  (None, 14, 14, 384)  3456       ['block_10_expand_relu[0][0]']   
 onv2D)                                                                                           
                                                                                                  
 block_10_depthwise_BN (BatchNo  (None, 14, 14, 384)  1536       ['block_10_depthwise[0][0]']     
 rmalization)                                                                                     
                                                                                                  
 block_10_depthwise_relu (ReLU)  (None, 14, 14, 384)  0          ['block_10_depthwise_BN[0][0]']  
                                                                                                  
 block_10_project (Conv2D)      (None, 14, 14, 96)   36864       ['block_10_depthwise_relu[0][0]']
                                                                                                  
 block_10_project_BN (BatchNorm  (None, 14, 14, 96)  384         ['block_10_project[0][0]']       
 alization)                                                                                       
                                                                                                  
 block_11_expand (Conv2D)       (None, 14, 14, 576)  55296       ['block_10_project_BN[0][0]']    
                                                                                                  
 block_11_expand_BN (BatchNorma  (None, 14, 14, 576)  2304       ['block_11_expand[0][0]']        
 lization)                                                                                        
                                                                                                  
 block_11_expand_relu (ReLU)    (None, 14, 14, 576)  0           ['block_11_expand_BN[0][0]']     
                                                                                                  
 block_11_depthwise (DepthwiseC  (None, 14, 14, 576)  5184       ['block_11_expand_relu[0][0]']   
 onv2D)                                                                                           
                                                                                                  
 block_11_depthwise_BN (BatchNo  (None, 14, 14, 576)  2304       ['block_11_depthwise[0][0]']     
 rmalization)                                                                                     
                                                                                                  
 block_11_depthwise_relu (ReLU)  (None, 14, 14, 576)  0          ['block_11_depthwise_BN[0][0]']  
                                                                                                  
 block_11_project (Conv2D)      (None, 14, 14, 96)   55296       ['block_11_depthwise_relu[0][0]']
                                                                                                  
 block_11_project_BN (BatchNorm  (None, 14, 14, 96)  384         ['block_11_project[0][0]']       
 alization)                                                                                       
                                                                                                  
 block_11_add (Add)             (None, 14, 14, 96)   0           ['block_10_project_BN[0][0]',    
                                                                  'block_11_project_BN[0][0]']    
                                                                                                  
 block_12_expand (Conv2D)       (None, 14, 14, 576)  55296       ['block_11_add[0][0]']           
                                                                                                  
 block_12_expand_BN (BatchNorma  (None, 14, 14, 576)  2304       ['block_12_expand[0][0]']        
 lization)                                                                                        
                                                                                                  
 block_12_expand_relu (ReLU)    (None, 14, 14, 576)  0           ['block_12_expand_BN[0][0]']     
                                                                                                  
 block_12_depthwise (DepthwiseC  (None, 14, 14, 576)  5184       ['block_12_expand_relu[0][0]']   
 onv2D)                                                                                           
                                                                                                  
 block_12_depthwise_BN (BatchNo  (None, 14, 14, 576)  2304       ['block_12_depthwise[0][0]']     
 rmalization)                                                                                     
                                                                                                  
 block_12_depthwise_relu (ReLU)  (None, 14, 14, 576)  0          ['block_12_depthwise_BN[0][0]']  
                                                                                                  
 block_12_project (Conv2D)      (None, 14, 14, 96)   55296       ['block_12_depthwise_relu[0][0]']
                                                                                                  
 block_12_project_BN (BatchNorm  (None, 14, 14, 96)  384         ['block_12_project[0][0]']       
 alization)                                                                                       
                                                                                                  
 block_12_add (Add)             (None, 14, 14, 96)   0           ['block_11_add[0][0]',           
                                                                  'block_12_project_BN[0][0]']    
                                                                                                  
 block_13_expand (Conv2D)       (None, 14, 14, 576)  55296       ['block_12_add[0][0]']           
                                                                                                  
 block_13_expand_BN (BatchNorma  (None, 14, 14, 576)  2304       ['block_13_expand[0][0]']        
 lization)                                                                                        
                                                                                                  
 block_13_expand_relu (ReLU)    (None, 14, 14, 576)  0           ['block_13_expand_BN[0][0]']     
                                                                                                  
 block_13_pad (ZeroPadding2D)   (None, 15, 15, 576)  0           ['block_13_expand_relu[0][0]']   
                                                                                                  
 block_13_depthwise (DepthwiseC  (None, 7, 7, 576)   5184        ['block_13_pad[0][0]']           
 onv2D)                                                                                           
                                                                                                  
 block_13_depthwise_BN (BatchNo  (None, 7, 7, 576)   2304        ['block_13_depthwise[0][0]']     
 rmalization)                                                                                     
                                                                                                  
 block_13_depthwise_relu (ReLU)  (None, 7, 7, 576)   0           ['block_13_depthwise_BN[0][0]']  
                                                                                                  
 block_13_project (Conv2D)      (None, 7, 7, 160)    92160       ['block_13_depthwise_relu[0][0]']
                                                                                                  
 block_13_project_BN (BatchNorm  (None, 7, 7, 160)   640         ['block_13_project[0][0]']       
 alization)                                                                                       
                                                                                                  
 block_14_expand (Conv2D)       (None, 7, 7, 960)    153600      ['block_13_project_BN[0][0]']    
                                                                                                  
 block_14_expand_BN (BatchNorma  (None, 7, 7, 960)   3840        ['block_14_expand[0][0]']        
 lization)                                                                                        
                                                                                                  
 block_14_expand_relu (ReLU)    (None, 7, 7, 960)    0           ['block_14_expand_BN[0][0]']     
                                                                                                  
 block_14_depthwise (DepthwiseC  (None, 7, 7, 960)   8640        ['block_14_expand_relu[0][0]']   
 onv2D)                                                                                           
                                                                                                  
 block_14_depthwise_BN (BatchNo  (None, 7, 7, 960)   3840        ['block_14_depthwise[0][0]']     
 rmalization)                                                                                     
                                                                                                  
 block_14_depthwise_relu (ReLU)  (None, 7, 7, 960)   0           ['block_14_depthwise_BN[0][0]']  
                                                                                                  
 block_14_project (Conv2D)      (None, 7, 7, 160)    153600      ['block_14_depthwise_relu[0][0]']
                                                                                                  
 block_14_project_BN (BatchNorm  (None, 7, 7, 160)   640         ['block_14_project[0][0]']       
 alization)                                                                                       
                                                                                                  
 block_14_add (Add)             (None, 7, 7, 160)    0           ['block_13_project_BN[0][0]',    
                                                                  'block_14_project_BN[0][0]']    
                                                                                                  
 block_15_expand (Conv2D)       (None, 7, 7, 960)    153600      ['block_14_add[0][0]']           
                                                                                                  
 block_15_expand_BN (BatchNorma  (None, 7, 7, 960)   3840        ['block_15_expand[0][0]']        
 lization)                                                                                        
                                                                                                  
 block_15_expand_relu (ReLU)    (None, 7, 7, 960)    0           ['block_15_expand_BN[0][0]']     
                                                                                                  
 block_15_depthwise (DepthwiseC  (None, 7, 7, 960)   8640        ['block_15_expand_relu[0][0]']   
 onv2D)                                                                                           
                                                                                                  
 block_15_depthwise_BN (BatchNo  (None, 7, 7, 960)   3840        ['block_15_depthwise[0][0]']     
 rmalization)                                                                                     
                                                                                                  
 block_15_depthwise_relu (ReLU)  (None, 7, 7, 960)   0           ['block_15_depthwise_BN[0][0]']  
                                                                                                  
 block_15_project (Conv2D)      (None, 7, 7, 160)    153600      ['block_15_depthwise_relu[0][0]']
                                                                                                  
 block_15_project_BN (BatchNorm  (None, 7, 7, 160)   640         ['block_15_project[0][0]']       
 alization)                                                                                       
                                                                                                  
 block_15_add (Add)             (None, 7, 7, 160)    0           ['block_14_add[0][0]',           
                                                                  'block_15_project_BN[0][0]']    
                                                                                                  
 block_16_expand (Conv2D)       (None, 7, 7, 960)    153600      ['block_15_add[0][0]']           
                                                                                                  
 block_16_expand_BN (BatchNorma  (None, 7, 7, 960)   3840        ['block_16_expand[0][0]']        
 lization)                                                                                        
                                                                                                  
 block_16_expand_relu (ReLU)    (None, 7, 7, 960)    0           ['block_16_expand_BN[0][0]']     
                                                                                                  
 block_16_depthwise (DepthwiseC  (None, 7, 7, 960)   8640        ['block_16_expand_relu[0][0]']   
 onv2D)                                                                                           
                                                                                                  
 block_16_depthwise_BN (BatchNo  (None, 7, 7, 960)   3840        ['block_16_depthwise[0][0]']     
 rmalization)                                                                                     
                                                                                                  
 block_16_depthwise_relu (ReLU)  (None, 7, 7, 960)   0           ['block_16_depthwise_BN[0][0]']  
                                                                                                  
 block_16_project (Conv2D)      (None, 7, 7, 320)    307200      ['block_16_depthwise_relu[0][0]']
                                                                                                  
 block_16_project_BN (BatchNorm  (None, 7, 7, 320)   1280        ['block_16_project[0][0]']       
 alization)                                                                                       
                                                                                                  
 Conv_1 (Conv2D)                (None, 7, 7, 1280)   409600      ['block_16_project_BN[0][0]']    
                                                                                                  
 Conv_1_bn (BatchNormalization)  (None, 7, 7, 1280)  5120        ['Conv_1[0][0]']                 
                                                                                                  
 out_relu (ReLU)                (None, 7, 7, 1280)   0           ['Conv_1_bn[0][0]']              
                                                                                                  
==================================================================================================
Total params: 2,257,984
Trainable params: 0
Non-trainable params: 2,257,984
__________________________________________________________________________________________________
Model: "model"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_2 (InputLayer)        [(None, 224, 224, 3)]     0         
                                                                 
 sequential (Sequential)     (None, 224, 224, 3)       0         
                                                                 
 tf.math.truediv (TFOpLambda  (None, 224, 224, 3)      0         
 )                                                               
                                                                 
 tf.math.subtract (TFOpLambd  (None, 224, 224, 3)      0         
 a)                                                              
                                                                 
 mobilenetv2_1.00_224 (Funct  (None, 7, 7, 1280)       2257984   
 ional)                                                          
                                                                 
 global_average_pooling2d (G  (None, 1280)             0         
 lobalAveragePooling2D)                                          
                                                                 
 dropout (Dropout)           (None, 1280)              0         
                                                                 
 dense (Dense)               (None, 1)                 1281      
                                                                 
=================================================================
Total params: 2,259,265
Trainable params: 1,281
Non-trainable params: 2,257,984
_________________________________________________________________
Epoch 1/30
2023-03-30 17:35:33.148115: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:113] Plugin optimizer for device_type GPU is enabled.
163/163 [==============================] - ETA: 0s - loss: 0.7097 - accuracy: 0.57272023-03-30 17:35:54.773311: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:113] Plugin optimizer for device_type GPU is enabled.
163/163 [==============================] - 23s 131ms/step - loss: 0.7097 - accuracy: 0.5727 - val_loss: 0.9480 - val_accuracy: 0.5000
Epoch 2/30
163/163 [==============================] - 21s 131ms/step - loss: 0.5251 - accuracy: 0.7542 - val_loss: 0.8245 - val_accuracy: 0.6875
Epoch 3/30
163/163 [==============================] - 21s 131ms/step - loss: 0.4419 - accuracy: 0.8179 - val_loss: 0.7328 - val_accuracy: 0.6875
Epoch 4/30
163/163 [==============================] - 22s 134ms/step - loss: 0.3828 - accuracy: 0.8570 - val_loss: 0.6522 - val_accuracy: 0.6875
Epoch 5/30
163/163 [==============================] - 22s 135ms/step - loss: 0.3388 - accuracy: 0.8827 - val_loss: 0.6093 - val_accuracy: 0.6875
Epoch 6/30
163/163 [==============================] - 21s 132ms/step - loss: 0.3081 - accuracy: 0.8938 - val_loss: 0.5684 - val_accuracy: 0.6875
Epoch 7/30
163/163 [==============================] - 23s 141ms/step - loss: 0.2828 - accuracy: 0.8990 - val_loss: 0.5242 - val_accuracy: 0.7500
Epoch 8/30
163/163 [==============================] - 22s 136ms/step - loss: 0.2637 - accuracy: 0.9022 - val_loss: 0.5086 - val_accuracy: 0.7500
Epoch 9/30
163/163 [==============================] - 22s 136ms/step - loss: 0.2473 - accuracy: 0.9072 - val_loss: 0.4737 - val_accuracy: 0.7500
Epoch 10/30
163/163 [==============================] - 23s 139ms/step - loss: 0.2296 - accuracy: 0.9156 - val_loss: 0.4629 - val_accuracy: 0.7500
Epoch 11/30
163/163 [==============================] - 22s 135ms/step - loss: 0.2179 - accuracy: 0.9162 - val_loss: 0.4307 - val_accuracy: 0.7500
Epoch 12/30
163/163 [==============================] - 22s 137ms/step - loss: 0.2063 - accuracy: 0.9191 - val_loss: 0.4183 - val_accuracy: 0.7500
Epoch 13/30
163/163 [==============================] - 22s 137ms/step - loss: 0.2001 - accuracy: 0.9235 - val_loss: 0.4075 - val_accuracy: 0.7500
Epoch 14/30
163/163 [==============================] - 22s 134ms/step - loss: 0.1905 - accuracy: 0.9231 - val_loss: 0.4052 - val_accuracy: 0.7500
Epoch 15/30
163/163 [==============================] - 22s 136ms/step - loss: 0.1840 - accuracy: 0.9210 - val_loss: 0.4011 - val_accuracy: 0.7500
Epoch 16/30
163/163 [==============================] - 22s 137ms/step - loss: 0.1735 - accuracy: 0.9317 - val_loss: 0.3768 - val_accuracy: 0.7500
Epoch 17/30
163/163 [==============================] - 22s 136ms/step - loss: 0.1758 - accuracy: 0.9321 - val_loss: 0.3638 - val_accuracy: 0.7500
Epoch 18/30
163/163 [==============================] - 22s 138ms/step - loss: 0.1639 - accuracy: 0.9367 - val_loss: 0.3552 - val_accuracy: 0.8125
Epoch 19/30
163/163 [==============================] - 22s 135ms/step - loss: 0.1593 - accuracy: 0.9344 - val_loss: 0.3442 - val_accuracy: 0.8125
Epoch 20/30
163/163 [==============================] - 22s 136ms/step - loss: 0.1578 - accuracy: 0.9350 - val_loss: 0.3163 - val_accuracy: 0.8125
Epoch 21/30
163/163 [==============================] - 22s 136ms/step - loss: 0.1544 - accuracy: 0.9392 - val_loss: 0.3256 - val_accuracy: 0.8125
Epoch 22/30
163/163 [==============================] - 22s 133ms/step - loss: 0.1516 - accuracy: 0.9398 - val_loss: 0.3006 - val_accuracy: 0.8125
Epoch 23/30
163/163 [==============================] - 22s 134ms/step - loss: 0.1498 - accuracy: 0.9438 - val_loss: 0.2973 - val_accuracy: 0.8125
Epoch 24/30
163/163 [==============================] - 22s 134ms/step - loss: 0.1443 - accuracy: 0.9413 - val_loss: 0.2888 - val_accuracy: 0.8750
Epoch 25/30
163/163 [==============================] - 22s 134ms/step - loss: 0.1432 - accuracy: 0.9398 - val_loss: 0.2969 - val_accuracy: 0.8125
Epoch 26/30
163/163 [==============================] - 22s 133ms/step - loss: 0.1361 - accuracy: 0.9465 - val_loss: 0.3081 - val_accuracy: 0.8125
Epoch 27/30
163/163 [==============================] - 22s 135ms/step - loss: 0.1387 - accuracy: 0.9463 - val_loss: 0.2880 - val_accuracy: 0.8750
Epoch 28/30
163/163 [==============================] - 22s 135ms/step - loss: 0.1390 - accuracy: 0.9456 - val_loss: 0.2952 - val_accuracy: 0.8125
Epoch 29/30
163/163 [==============================] - 22s 138ms/step - loss: 0.1320 - accuracy: 0.9486 - val_loss: 0.2832 - val_accuracy: 0.8750
Epoch 30/30
163/163 [==============================] - 22s 135ms/step - loss: 0.1337 - accuracy: 0.9452 - val_loss: 0.3033 - val_accuracy: 0.8125
accuracy / loss on the test dataset: 0.8157051205635071 / 0.4630528390407562
```

The learning curves of the training and validation accuracy / loss looks as following:

![plot](./accuracy-loss.png)

Finally the created environment above can be removed through:

```
conda remove --name pneumonia --all
```

The achieved accuracy on the test dataset is `.815` as it can be spotted by the last line of the log output above.
