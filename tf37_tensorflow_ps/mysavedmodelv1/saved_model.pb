Íæ
ý
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
¾
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"serve*2.1.02unknown8¢ã

dense-128-relu_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*(
shared_namedense-128-relu_4/kernel

+dense-128-relu_4/kernel/Read/ReadVariableOpReadVariableOpdense-128-relu_4/kernel* 
_output_shapes
:
*
dtype0

dense-128-relu_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_namedense-128-relu_4/bias
|
)dense-128-relu_4/bias/Read/ReadVariableOpReadVariableOpdense-128-relu_4/bias*
_output_shapes	
:*
dtype0

dense-10-softmax_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	
**
shared_namedense-10-softmax_5/kernel

-dense-10-softmax_5/kernel/Read/ReadVariableOpReadVariableOpdense-10-softmax_5/kernel*
_output_shapes
:	
*
dtype0

dense-10-softmax_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*(
shared_namedense-10-softmax_5/bias

+dense-10-softmax_5/bias/Read/ReadVariableOpReadVariableOpdense-10-softmax_5/bias*
_output_shapes
:
*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0

Adam/dense-128-relu_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*/
shared_name Adam/dense-128-relu_4/kernel/m

2Adam/dense-128-relu_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense-128-relu_4/kernel/m* 
_output_shapes
:
*
dtype0

Adam/dense-128-relu_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameAdam/dense-128-relu_4/bias/m

0Adam/dense-128-relu_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense-128-relu_4/bias/m*
_output_shapes	
:*
dtype0

 Adam/dense-10-softmax_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	
*1
shared_name" Adam/dense-10-softmax_5/kernel/m

4Adam/dense-10-softmax_5/kernel/m/Read/ReadVariableOpReadVariableOp Adam/dense-10-softmax_5/kernel/m*
_output_shapes
:	
*
dtype0

Adam/dense-10-softmax_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*/
shared_name Adam/dense-10-softmax_5/bias/m

2Adam/dense-10-softmax_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense-10-softmax_5/bias/m*
_output_shapes
:
*
dtype0

Adam/dense-128-relu_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*/
shared_name Adam/dense-128-relu_4/kernel/v

2Adam/dense-128-relu_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense-128-relu_4/kernel/v* 
_output_shapes
:
*
dtype0

Adam/dense-128-relu_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameAdam/dense-128-relu_4/bias/v

0Adam/dense-128-relu_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense-128-relu_4/bias/v*
_output_shapes	
:*
dtype0

 Adam/dense-10-softmax_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	
*1
shared_name" Adam/dense-10-softmax_5/kernel/v

4Adam/dense-10-softmax_5/kernel/v/Read/ReadVariableOpReadVariableOp Adam/dense-10-softmax_5/kernel/v*
_output_shapes
:	
*
dtype0

Adam/dense-10-softmax_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*/
shared_name Adam/dense-10-softmax_5/bias/v

2Adam/dense-10-softmax_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense-10-softmax_5/bias/v*
_output_shapes
:
*
dtype0

NoOpNoOp
ï
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ª
value B B
Ù
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
	optimizer
trainable_variables
regularization_losses
	variables
		keras_api


signatures
 
R
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api

iter

beta_1

beta_2
	decay
learning_ratem<m=m>m?v@vAvBvC

0
1
2
3
 

0
1
2
3

 layer_regularization_losses
trainable_variables

!layers
regularization_losses
"metrics
	variables
#non_trainable_variables
 
 
 
 

$layer_regularization_losses
trainable_variables

%layers
regularization_losses
&metrics
	variables
'non_trainable_variables
ca
VARIABLE_VALUEdense-128-relu_4/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEdense-128-relu_4/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1

(layer_regularization_losses
trainable_variables

)layers
regularization_losses
*metrics
	variables
+non_trainable_variables
ec
VARIABLE_VALUEdense-10-softmax_5/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEdense-10-softmax_5/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1

,layer_regularization_losses
trainable_variables

-layers
regularization_losses
.metrics
	variables
/non_trainable_variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 

0
1
2

00
 
 
 
 
 
 
 
 
 
 
 
 
 
x
	1total
	2count
3
_fn_kwargs
4trainable_variables
5regularization_losses
6	variables
7	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 
 
 

10
21

8layer_regularization_losses
4trainable_variables

9layers
5regularization_losses
:metrics
6	variables
;non_trainable_variables
 
 
 

10
21

VARIABLE_VALUEAdam/dense-128-relu_4/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/dense-128-relu_4/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE Adam/dense-10-softmax_5/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/dense-10-softmax_5/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/dense-128-relu_4/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/dense-128-relu_4/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE Adam/dense-10-softmax_5/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/dense-10-softmax_5/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_flatten_5_inputPlaceholder*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0* 
shape:ÿÿÿÿÿÿÿÿÿ

StatefulPartitionedCallStatefulPartitionedCallserving_default_flatten_5_inputdense-128-relu_4/kerneldense-128-relu_4/biasdense-10-softmax_5/kerneldense-10-softmax_5/bias*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
**
config_proto

CPU

GPU 2J 8*.
f)R'
%__inference_signature_wrapper_2097350
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ì
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename+dense-128-relu_4/kernel/Read/ReadVariableOp)dense-128-relu_4/bias/Read/ReadVariableOp-dense-10-softmax_5/kernel/Read/ReadVariableOp+dense-10-softmax_5/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp2Adam/dense-128-relu_4/kernel/m/Read/ReadVariableOp0Adam/dense-128-relu_4/bias/m/Read/ReadVariableOp4Adam/dense-10-softmax_5/kernel/m/Read/ReadVariableOp2Adam/dense-10-softmax_5/bias/m/Read/ReadVariableOp2Adam/dense-128-relu_4/kernel/v/Read/ReadVariableOp0Adam/dense-128-relu_4/bias/v/Read/ReadVariableOp4Adam/dense-10-softmax_5/kernel/v/Read/ReadVariableOp2Adam/dense-10-softmax_5/bias/v/Read/ReadVariableOpConst* 
Tin
2	*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: **
config_proto

CPU

GPU 2J 8*)
f$R"
 __inference__traced_save_2097536
Ë
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense-128-relu_4/kerneldense-128-relu_4/biasdense-10-softmax_5/kerneldense-10-softmax_5/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/dense-128-relu_4/kernel/mAdam/dense-128-relu_4/bias/m Adam/dense-10-softmax_5/kernel/mAdam/dense-10-softmax_5/bias/mAdam/dense-128-relu_4/kernel/vAdam/dense-128-relu_4/bias/v Adam/dense-10-softmax_5/kernel/vAdam/dense-10-softmax_5/bias/v*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: **
config_proto

CPU

GPU 2J 8*,
f'R%
#__inference__traced_restore_2097605Í
T
í

#__inference__traced_restore_2097605
file_prefix,
(assignvariableop_dense_128_relu_4_kernel,
(assignvariableop_1_dense_128_relu_4_bias0
,assignvariableop_2_dense_10_softmax_5_kernel.
*assignvariableop_3_dense_10_softmax_5_bias 
assignvariableop_4_adam_iter"
assignvariableop_5_adam_beta_1"
assignvariableop_6_adam_beta_2!
assignvariableop_7_adam_decay)
%assignvariableop_8_adam_learning_rate
assignvariableop_9_total
assignvariableop_10_count6
2assignvariableop_11_adam_dense_128_relu_4_kernel_m4
0assignvariableop_12_adam_dense_128_relu_4_bias_m8
4assignvariableop_13_adam_dense_10_softmax_5_kernel_m6
2assignvariableop_14_adam_dense_10_softmax_5_bias_m6
2assignvariableop_15_adam_dense_128_relu_4_kernel_v4
0assignvariableop_16_adam_dense_128_relu_4_bias_v8
4assignvariableop_17_adam_dense_10_softmax_5_kernel_v6
2assignvariableop_18_adam_dense_10_softmax_5_bias_v
identity_20¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_2¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9¢	RestoreV2¢RestoreV2_1Î

RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ú	
valueÐ	BÍ	B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names´
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*9
value0B.B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*`
_output_shapesN
L:::::::::::::::::::*!
dtypes
2	2
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOp(assignvariableop_dense_128_relu_4_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1
AssignVariableOp_1AssignVariableOp(assignvariableop_1_dense_128_relu_4_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2¢
AssignVariableOp_2AssignVariableOp,assignvariableop_2_dense_10_softmax_5_kernelIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3 
AssignVariableOp_3AssignVariableOp*assignvariableop_3_dense_10_softmax_5_biasIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0	*
_output_shapes
:2

Identity_4
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_iterIdentity_4:output:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_beta_1Identity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_2Identity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_decayIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8
AssignVariableOp_8AssignVariableOp%assignvariableop_8_adam_learning_rateIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9
AssignVariableOp_9AssignVariableOpassignvariableop_9_totalIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10
AssignVariableOp_10AssignVariableOpassignvariableop_10_countIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11«
AssignVariableOp_11AssignVariableOp2assignvariableop_11_adam_dense_128_relu_4_kernel_mIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12©
AssignVariableOp_12AssignVariableOp0assignvariableop_12_adam_dense_128_relu_4_bias_mIdentity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13­
AssignVariableOp_13AssignVariableOp4assignvariableop_13_adam_dense_10_softmax_5_kernel_mIdentity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14«
AssignVariableOp_14AssignVariableOp2assignvariableop_14_adam_dense_10_softmax_5_bias_mIdentity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15«
AssignVariableOp_15AssignVariableOp2assignvariableop_15_adam_dense_128_relu_4_kernel_vIdentity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16©
AssignVariableOp_16AssignVariableOp0assignvariableop_16_adam_dense_128_relu_4_bias_vIdentity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17­
AssignVariableOp_17AssignVariableOp4assignvariableop_17_adam_dense_10_softmax_5_kernel_vIdentity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18«
AssignVariableOp_18AssignVariableOp2assignvariableop_18_adam_dense_10_softmax_5_bias_vIdentity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18¨
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_names
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slicesÄ
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp
Identity_19Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_19
Identity_20IdentityIdentity_19:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_20"#
identity_20Identity_20:output:0*a
_input_shapesP
N: :::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:+ '
%
_user_specified_namefile_prefix
Ä

I__inference_sequential_5_layer_call_and_return_conditional_losses_2097305

inputs1
-dense_128_relu_statefulpartitionedcall_args_11
-dense_128_relu_statefulpartitionedcall_args_23
/dense_10_softmax_statefulpartitionedcall_args_13
/dense_10_softmax_statefulpartitionedcall_args_2
identity¢(dense-10-softmax/StatefulPartitionedCall¢&dense-128-relu/StatefulPartitionedCallÃ
flatten_5/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_flatten_5_layer_call_and_return_conditional_losses_20972252
flatten_5/PartitionedCallæ
&dense-128-relu/StatefulPartitionedCallStatefulPartitionedCall"flatten_5/PartitionedCall:output:0-dense_128_relu_statefulpartitionedcall_args_1-dense_128_relu_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*T
fORM
K__inference_dense-128-relu_layer_call_and_return_conditional_losses_20972442(
&dense-128-relu/StatefulPartitionedCallü
(dense-10-softmax/StatefulPartitionedCallStatefulPartitionedCall/dense-128-relu/StatefulPartitionedCall:output:0/dense_10_softmax_statefulpartitionedcall_args_1/dense_10_softmax_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
**
config_proto

CPU

GPU 2J 8*V
fQRO
M__inference_dense-10-softmax_layer_call_and_return_conditional_losses_20972672*
(dense-10-softmax/StatefulPartitionedCallÙ
IdentityIdentity1dense-10-softmax/StatefulPartitionedCall:output:0)^dense-10-softmax/StatefulPartitionedCall'^dense-128-relu/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ::::2T
(dense-10-softmax/StatefulPartitionedCall(dense-10-softmax/StatefulPartitionedCall2P
&dense-128-relu/StatefulPartitionedCall&dense-128-relu/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
ý
÷
%__inference_signature_wrapper_2097350
flatten_5_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity¢StatefulPartitionedCall¯
StatefulPartitionedCallStatefulPartitionedCallflatten_5_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
**
config_proto

CPU

GPU 2J 8*+
f&R$
"__inference__wrapped_model_20972152
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:/ +
)
_user_specified_nameflatten_5_input
­

.__inference_sequential_5_layer_call_fn_2097312
flatten_5_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity¢StatefulPartitionedCallÖ
StatefulPartitionedCallStatefulPartitionedCallflatten_5_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
**
config_proto

CPU

GPU 2J 8*R
fMRK
I__inference_sequential_5_layer_call_and_return_conditional_losses_20973052
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:/ +
)
_user_specified_nameflatten_5_input

÷
.__inference_sequential_5_layer_call_fn_2097399

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity¢StatefulPartitionedCallÍ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
**
config_proto

CPU

GPU 2J 8*R
fMRK
I__inference_sequential_5_layer_call_and_return_conditional_losses_20973052
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
ß

I__inference_sequential_5_layer_call_and_return_conditional_losses_2097280
flatten_5_input1
-dense_128_relu_statefulpartitionedcall_args_11
-dense_128_relu_statefulpartitionedcall_args_23
/dense_10_softmax_statefulpartitionedcall_args_13
/dense_10_softmax_statefulpartitionedcall_args_2
identity¢(dense-10-softmax/StatefulPartitionedCall¢&dense-128-relu/StatefulPartitionedCallÌ
flatten_5/PartitionedCallPartitionedCallflatten_5_input*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_flatten_5_layer_call_and_return_conditional_losses_20972252
flatten_5/PartitionedCallæ
&dense-128-relu/StatefulPartitionedCallStatefulPartitionedCall"flatten_5/PartitionedCall:output:0-dense_128_relu_statefulpartitionedcall_args_1-dense_128_relu_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*T
fORM
K__inference_dense-128-relu_layer_call_and_return_conditional_losses_20972442(
&dense-128-relu/StatefulPartitionedCallü
(dense-10-softmax/StatefulPartitionedCallStatefulPartitionedCall/dense-128-relu/StatefulPartitionedCall:output:0/dense_10_softmax_statefulpartitionedcall_args_1/dense_10_softmax_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
**
config_proto

CPU

GPU 2J 8*V
fQRO
M__inference_dense-10-softmax_layer_call_and_return_conditional_losses_20972672*
(dense-10-softmax/StatefulPartitionedCallÙ
IdentityIdentity1dense-10-softmax/StatefulPartitionedCall:output:0)^dense-10-softmax/StatefulPartitionedCall'^dense-128-relu/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ::::2T
(dense-10-softmax/StatefulPartitionedCall(dense-10-softmax/StatefulPartitionedCall2P
&dense-128-relu/StatefulPartitionedCall&dense-128-relu/StatefulPartitionedCall:/ +
)
_user_specified_nameflatten_5_input

³
2__inference_dense-10-softmax_layer_call_fn_2097455

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
**
config_proto

CPU

GPU 2J 8*V
fQRO
M__inference_dense-10-softmax_layer_call_and_return_conditional_losses_20972672
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs

b
F__inference_flatten_5_layer_call_and_return_conditional_losses_2097414

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:& "
 
_user_specified_nameinputs
Ä

I__inference_sequential_5_layer_call_and_return_conditional_losses_2097325

inputs1
-dense_128_relu_statefulpartitionedcall_args_11
-dense_128_relu_statefulpartitionedcall_args_23
/dense_10_softmax_statefulpartitionedcall_args_13
/dense_10_softmax_statefulpartitionedcall_args_2
identity¢(dense-10-softmax/StatefulPartitionedCall¢&dense-128-relu/StatefulPartitionedCallÃ
flatten_5/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_flatten_5_layer_call_and_return_conditional_losses_20972252
flatten_5/PartitionedCallæ
&dense-128-relu/StatefulPartitionedCallStatefulPartitionedCall"flatten_5/PartitionedCall:output:0-dense_128_relu_statefulpartitionedcall_args_1-dense_128_relu_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*T
fORM
K__inference_dense-128-relu_layer_call_and_return_conditional_losses_20972442(
&dense-128-relu/StatefulPartitionedCallü
(dense-10-softmax/StatefulPartitionedCallStatefulPartitionedCall/dense-128-relu/StatefulPartitionedCall:output:0/dense_10_softmax_statefulpartitionedcall_args_1/dense_10_softmax_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
**
config_proto

CPU

GPU 2J 8*V
fQRO
M__inference_dense-10-softmax_layer_call_and_return_conditional_losses_20972672*
(dense-10-softmax/StatefulPartitionedCallÙ
IdentityIdentity1dense-10-softmax/StatefulPartitionedCall:output:0)^dense-10-softmax/StatefulPartitionedCall'^dense-128-relu/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ::::2T
(dense-10-softmax/StatefulPartitionedCall(dense-10-softmax/StatefulPartitionedCall2P
&dense-128-relu/StatefulPartitionedCall&dense-128-relu/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
­

.__inference_sequential_5_layer_call_fn_2097332
flatten_5_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity¢StatefulPartitionedCallÖ
StatefulPartitionedCallStatefulPartitionedCallflatten_5_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
**
config_proto

CPU

GPU 2J 8*R
fMRK
I__inference_sequential_5_layer_call_and_return_conditional_losses_20973252
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:/ +
)
_user_specified_nameflatten_5_input

±
0__inference_dense-128-relu_layer_call_fn_2097437

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*T
fORM
K__inference_dense-128-relu_layer_call_and_return_conditional_losses_20972442
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs

÷
.__inference_sequential_5_layer_call_fn_2097408

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity¢StatefulPartitionedCallÍ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
**
config_proto

CPU

GPU 2J 8*R
fMRK
I__inference_sequential_5_layer_call_and_return_conditional_losses_20973252
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs

b
F__inference_flatten_5_layer_call_and_return_conditional_losses_2097225

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:& "
 
_user_specified_nameinputs
Ô	
ä
K__inference_dense-128-relu_layer_call_and_return_conditional_losses_2097244

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
Ø	
æ
M__inference_dense-10-softmax_layer_call_and_return_conditional_losses_2097448

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2	
Softmax
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
æ
Ü
I__inference_sequential_5_layer_call_and_return_conditional_losses_2097390

inputs1
-dense_128_relu_matmul_readvariableop_resource2
.dense_128_relu_biasadd_readvariableop_resource3
/dense_10_softmax_matmul_readvariableop_resource4
0dense_10_softmax_biasadd_readvariableop_resource
identity¢'dense-10-softmax/BiasAdd/ReadVariableOp¢&dense-10-softmax/MatMul/ReadVariableOp¢%dense-128-relu/BiasAdd/ReadVariableOp¢$dense-128-relu/MatMul/ReadVariableOps
flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2
flatten_5/Const
flatten_5/ReshapeReshapeinputsflatten_5/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
flatten_5/Reshape¼
$dense-128-relu/MatMul/ReadVariableOpReadVariableOp-dense_128_relu_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02&
$dense-128-relu/MatMul/ReadVariableOpµ
dense-128-relu/MatMulMatMulflatten_5/Reshape:output:0,dense-128-relu/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense-128-relu/MatMulº
%dense-128-relu/BiasAdd/ReadVariableOpReadVariableOp.dense_128_relu_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02'
%dense-128-relu/BiasAdd/ReadVariableOp¾
dense-128-relu/BiasAddBiasAdddense-128-relu/MatMul:product:0-dense-128-relu/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense-128-relu/BiasAdd
dense-128-relu/ReluReludense-128-relu/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense-128-relu/ReluÁ
&dense-10-softmax/MatMul/ReadVariableOpReadVariableOp/dense_10_softmax_matmul_readvariableop_resource*
_output_shapes
:	
*
dtype02(
&dense-10-softmax/MatMul/ReadVariableOpÁ
dense-10-softmax/MatMulMatMul!dense-128-relu/Relu:activations:0.dense-10-softmax/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense-10-softmax/MatMul¿
'dense-10-softmax/BiasAdd/ReadVariableOpReadVariableOp0dense_10_softmax_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02)
'dense-10-softmax/BiasAdd/ReadVariableOpÅ
dense-10-softmax/BiasAddBiasAdd!dense-10-softmax/MatMul:product:0/dense-10-softmax/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense-10-softmax/BiasAdd
dense-10-softmax/SoftmaxSoftmax!dense-10-softmax/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense-10-softmax/Softmax
IdentityIdentity"dense-10-softmax/Softmax:softmax:0(^dense-10-softmax/BiasAdd/ReadVariableOp'^dense-10-softmax/MatMul/ReadVariableOp&^dense-128-relu/BiasAdd/ReadVariableOp%^dense-128-relu/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ::::2R
'dense-10-softmax/BiasAdd/ReadVariableOp'dense-10-softmax/BiasAdd/ReadVariableOp2P
&dense-10-softmax/MatMul/ReadVariableOp&dense-10-softmax/MatMul/ReadVariableOp2N
%dense-128-relu/BiasAdd/ReadVariableOp%dense-128-relu/BiasAdd/ReadVariableOp2L
$dense-128-relu/MatMul/ReadVariableOp$dense-128-relu/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
æ
Ü
I__inference_sequential_5_layer_call_and_return_conditional_losses_2097370

inputs1
-dense_128_relu_matmul_readvariableop_resource2
.dense_128_relu_biasadd_readvariableop_resource3
/dense_10_softmax_matmul_readvariableop_resource4
0dense_10_softmax_biasadd_readvariableop_resource
identity¢'dense-10-softmax/BiasAdd/ReadVariableOp¢&dense-10-softmax/MatMul/ReadVariableOp¢%dense-128-relu/BiasAdd/ReadVariableOp¢$dense-128-relu/MatMul/ReadVariableOps
flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2
flatten_5/Const
flatten_5/ReshapeReshapeinputsflatten_5/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
flatten_5/Reshape¼
$dense-128-relu/MatMul/ReadVariableOpReadVariableOp-dense_128_relu_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02&
$dense-128-relu/MatMul/ReadVariableOpµ
dense-128-relu/MatMulMatMulflatten_5/Reshape:output:0,dense-128-relu/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense-128-relu/MatMulº
%dense-128-relu/BiasAdd/ReadVariableOpReadVariableOp.dense_128_relu_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02'
%dense-128-relu/BiasAdd/ReadVariableOp¾
dense-128-relu/BiasAddBiasAdddense-128-relu/MatMul:product:0-dense-128-relu/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense-128-relu/BiasAdd
dense-128-relu/ReluReludense-128-relu/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense-128-relu/ReluÁ
&dense-10-softmax/MatMul/ReadVariableOpReadVariableOp/dense_10_softmax_matmul_readvariableop_resource*
_output_shapes
:	
*
dtype02(
&dense-10-softmax/MatMul/ReadVariableOpÁ
dense-10-softmax/MatMulMatMul!dense-128-relu/Relu:activations:0.dense-10-softmax/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense-10-softmax/MatMul¿
'dense-10-softmax/BiasAdd/ReadVariableOpReadVariableOp0dense_10_softmax_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02)
'dense-10-softmax/BiasAdd/ReadVariableOpÅ
dense-10-softmax/BiasAddBiasAdd!dense-10-softmax/MatMul:product:0/dense-10-softmax/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense-10-softmax/BiasAdd
dense-10-softmax/SoftmaxSoftmax!dense-10-softmax/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense-10-softmax/Softmax
IdentityIdentity"dense-10-softmax/Softmax:softmax:0(^dense-10-softmax/BiasAdd/ReadVariableOp'^dense-10-softmax/MatMul/ReadVariableOp&^dense-128-relu/BiasAdd/ReadVariableOp%^dense-128-relu/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ::::2R
'dense-10-softmax/BiasAdd/ReadVariableOp'dense-10-softmax/BiasAdd/ReadVariableOp2P
&dense-10-softmax/MatMul/ReadVariableOp&dense-10-softmax/MatMul/ReadVariableOp2N
%dense-128-relu/BiasAdd/ReadVariableOp%dense-128-relu/BiasAdd/ReadVariableOp2L
$dense-128-relu/MatMul/ReadVariableOp$dense-128-relu/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
²1
	
 __inference__traced_save_2097536
file_prefix6
2savev2_dense_128_relu_4_kernel_read_readvariableop4
0savev2_dense_128_relu_4_bias_read_readvariableop8
4savev2_dense_10_softmax_5_kernel_read_readvariableop6
2savev2_dense_10_softmax_5_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop=
9savev2_adam_dense_128_relu_4_kernel_m_read_readvariableop;
7savev2_adam_dense_128_relu_4_bias_m_read_readvariableop?
;savev2_adam_dense_10_softmax_5_kernel_m_read_readvariableop=
9savev2_adam_dense_10_softmax_5_bias_m_read_readvariableop=
9savev2_adam_dense_128_relu_4_kernel_v_read_readvariableop;
7savev2_adam_dense_128_relu_4_bias_v_read_readvariableop?
;savev2_adam_dense_10_softmax_5_kernel_v_read_readvariableop=
9savev2_adam_dense_10_softmax_5_bias_v_read_readvariableop
savev2_1_const

identity_1¢MergeV2Checkpoints¢SaveV2¢SaveV2_1¥
StringJoin/inputs_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_7f0329fdca3849f0871ab511683cabee/part2
StringJoin/inputs_1

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameÈ

SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ú	
valueÐ	BÍ	B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names®
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*9
value0B.B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesö
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:02savev2_dense_128_relu_4_kernel_read_readvariableop0savev2_dense_128_relu_4_bias_read_readvariableop4savev2_dense_10_softmax_5_kernel_read_readvariableop2savev2_dense_10_softmax_5_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop9savev2_adam_dense_128_relu_4_kernel_m_read_readvariableop7savev2_adam_dense_128_relu_4_bias_m_read_readvariableop;savev2_adam_dense_10_softmax_5_kernel_m_read_readvariableop9savev2_adam_dense_10_softmax_5_bias_m_read_readvariableop9savev2_adam_dense_128_relu_4_kernel_v_read_readvariableop7savev2_adam_dense_128_relu_4_bias_v_read_readvariableop;savev2_adam_dense_10_softmax_5_kernel_v_read_readvariableop9savev2_adam_dense_10_softmax_5_bias_v_read_readvariableop"/device:CPU:0*
_output_shapes
 *!
dtypes
2	2
SaveV2
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shard¬
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1¢
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slicesÏ
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1ã
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¬
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
~: :
::	
:
: : : : : : : :
::	
:
:
::	
:
: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:+ '
%
_user_specified_namefile_prefix
Ø	
æ
M__inference_dense-10-softmax_layer_call_and_return_conditional_losses_2097267

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2	
Softmax
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
ç
¦
"__inference__wrapped_model_2097215
flatten_5_input>
:sequential_5_dense_128_relu_matmul_readvariableop_resource?
;sequential_5_dense_128_relu_biasadd_readvariableop_resource@
<sequential_5_dense_10_softmax_matmul_readvariableop_resourceA
=sequential_5_dense_10_softmax_biasadd_readvariableop_resource
identity¢4sequential_5/dense-10-softmax/BiasAdd/ReadVariableOp¢3sequential_5/dense-10-softmax/MatMul/ReadVariableOp¢2sequential_5/dense-128-relu/BiasAdd/ReadVariableOp¢1sequential_5/dense-128-relu/MatMul/ReadVariableOp
sequential_5/flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2
sequential_5/flatten_5/Const¶
sequential_5/flatten_5/ReshapeReshapeflatten_5_input%sequential_5/flatten_5/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
sequential_5/flatten_5/Reshapeã
1sequential_5/dense-128-relu/MatMul/ReadVariableOpReadVariableOp:sequential_5_dense_128_relu_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype023
1sequential_5/dense-128-relu/MatMul/ReadVariableOpé
"sequential_5/dense-128-relu/MatMulMatMul'sequential_5/flatten_5/Reshape:output:09sequential_5/dense-128-relu/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"sequential_5/dense-128-relu/MatMulá
2sequential_5/dense-128-relu/BiasAdd/ReadVariableOpReadVariableOp;sequential_5_dense_128_relu_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype024
2sequential_5/dense-128-relu/BiasAdd/ReadVariableOpò
#sequential_5/dense-128-relu/BiasAddBiasAdd,sequential_5/dense-128-relu/MatMul:product:0:sequential_5/dense-128-relu/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#sequential_5/dense-128-relu/BiasAdd­
 sequential_5/dense-128-relu/ReluRelu,sequential_5/dense-128-relu/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 sequential_5/dense-128-relu/Reluè
3sequential_5/dense-10-softmax/MatMul/ReadVariableOpReadVariableOp<sequential_5_dense_10_softmax_matmul_readvariableop_resource*
_output_shapes
:	
*
dtype025
3sequential_5/dense-10-softmax/MatMul/ReadVariableOpõ
$sequential_5/dense-10-softmax/MatMulMatMul.sequential_5/dense-128-relu/Relu:activations:0;sequential_5/dense-10-softmax/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2&
$sequential_5/dense-10-softmax/MatMulæ
4sequential_5/dense-10-softmax/BiasAdd/ReadVariableOpReadVariableOp=sequential_5_dense_10_softmax_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype026
4sequential_5/dense-10-softmax/BiasAdd/ReadVariableOpù
%sequential_5/dense-10-softmax/BiasAddBiasAdd.sequential_5/dense-10-softmax/MatMul:product:0<sequential_5/dense-10-softmax/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2'
%sequential_5/dense-10-softmax/BiasAdd»
%sequential_5/dense-10-softmax/SoftmaxSoftmax.sequential_5/dense-10-softmax/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2'
%sequential_5/dense-10-softmax/SoftmaxÙ
IdentityIdentity/sequential_5/dense-10-softmax/Softmax:softmax:05^sequential_5/dense-10-softmax/BiasAdd/ReadVariableOp4^sequential_5/dense-10-softmax/MatMul/ReadVariableOp3^sequential_5/dense-128-relu/BiasAdd/ReadVariableOp2^sequential_5/dense-128-relu/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ::::2l
4sequential_5/dense-10-softmax/BiasAdd/ReadVariableOp4sequential_5/dense-10-softmax/BiasAdd/ReadVariableOp2j
3sequential_5/dense-10-softmax/MatMul/ReadVariableOp3sequential_5/dense-10-softmax/MatMul/ReadVariableOp2h
2sequential_5/dense-128-relu/BiasAdd/ReadVariableOp2sequential_5/dense-128-relu/BiasAdd/ReadVariableOp2f
1sequential_5/dense-128-relu/MatMul/ReadVariableOp1sequential_5/dense-128-relu/MatMul/ReadVariableOp:/ +
)
_user_specified_nameflatten_5_input
ß

I__inference_sequential_5_layer_call_and_return_conditional_losses_2097291
flatten_5_input1
-dense_128_relu_statefulpartitionedcall_args_11
-dense_128_relu_statefulpartitionedcall_args_23
/dense_10_softmax_statefulpartitionedcall_args_13
/dense_10_softmax_statefulpartitionedcall_args_2
identity¢(dense-10-softmax/StatefulPartitionedCall¢&dense-128-relu/StatefulPartitionedCallÌ
flatten_5/PartitionedCallPartitionedCallflatten_5_input*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_flatten_5_layer_call_and_return_conditional_losses_20972252
flatten_5/PartitionedCallæ
&dense-128-relu/StatefulPartitionedCallStatefulPartitionedCall"flatten_5/PartitionedCall:output:0-dense_128_relu_statefulpartitionedcall_args_1-dense_128_relu_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*T
fORM
K__inference_dense-128-relu_layer_call_and_return_conditional_losses_20972442(
&dense-128-relu/StatefulPartitionedCallü
(dense-10-softmax/StatefulPartitionedCallStatefulPartitionedCall/dense-128-relu/StatefulPartitionedCall:output:0/dense_10_softmax_statefulpartitionedcall_args_1/dense_10_softmax_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
**
config_proto

CPU

GPU 2J 8*V
fQRO
M__inference_dense-10-softmax_layer_call_and_return_conditional_losses_20972672*
(dense-10-softmax/StatefulPartitionedCallÙ
IdentityIdentity1dense-10-softmax/StatefulPartitionedCall:output:0)^dense-10-softmax/StatefulPartitionedCall'^dense-128-relu/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ::::2T
(dense-10-softmax/StatefulPartitionedCall(dense-10-softmax/StatefulPartitionedCall2P
&dense-128-relu/StatefulPartitionedCall&dense-128-relu/StatefulPartitionedCall:/ +
)
_user_specified_nameflatten_5_input
Ý
G
+__inference_flatten_5_layer_call_fn_2097419

inputs
identity¯
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_flatten_5_layer_call_and_return_conditional_losses_20972252
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:& "
 
_user_specified_nameinputs
Ô	
ä
K__inference_dense-128-relu_layer_call_and_return_conditional_losses_2097430

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs"¯L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ç
serving_default³
O
flatten_5_input<
!serving_default_flatten_5_input:0ÿÿÿÿÿÿÿÿÿD
dense-10-softmax0
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿ
tensorflow/serving/predict:|
ï
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
	optimizer
trainable_variables
regularization_losses
	variables
		keras_api


signatures
D__call__
E_default_save_signature
*F&call_and_return_all_conditional_losses"¼
_tf_keras_sequential{"class_name": "Sequential", "name": "sequential_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "sequential_5", "layers": [{"class_name": "Flatten", "config": {"name": "flatten_5", "trainable": true, "batch_input_shape": [null, 28, 28], "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense-128-relu", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense-10-softmax", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}, "is_graph_network": true, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_5", "layers": [{"class_name": "Flatten", "config": {"name": "flatten_5", "trainable": true, "batch_input_shape": [null, 28, 28], "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense-128-relu", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense-10-softmax", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "sparse_categorical_crossentropy", "metrics": ["accuracy"], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
·"´
_tf_keras_input_layer{"class_name": "InputLayer", "name": "flatten_5_input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": [null, 28, 28], "config": {"batch_input_shape": [null, 28, 28], "dtype": "float32", "sparse": false, "ragged": false, "name": "flatten_5_input"}}
ß
trainable_variables
regularization_losses
	variables
	keras_api
G__call__
*H&call_and_return_all_conditional_losses"Ð
_tf_keras_layer¶{"class_name": "Flatten", "name": "flatten_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": [null, 28, 28], "config": {"name": "flatten_5", "trainable": true, "batch_input_shape": [null, 28, 28], "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}


kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
I__call__
*J&call_and_return_all_conditional_losses"Ü
_tf_keras_layerÂ{"class_name": "Dense", "name": "dense-128-relu", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense-128-relu", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 784}}}}


kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
K__call__
*L&call_and_return_all_conditional_losses"â
_tf_keras_layerÈ{"class_name": "Dense", "name": "dense-10-softmax", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense-10-softmax", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}}

iter

beta_1

beta_2
	decay
learning_ratem<m=m>m?v@vAvBvC"
	optimizer
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
·
 layer_regularization_losses
trainable_variables

!layers
regularization_losses
"metrics
	variables
#non_trainable_variables
D__call__
E_default_save_signature
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses"
_generic_user_object
,
Mserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper

$layer_regularization_losses
trainable_variables

%layers
regularization_losses
&metrics
	variables
'non_trainable_variables
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses"
_generic_user_object
+:)
2dense-128-relu_4/kernel
$:"2dense-128-relu_4/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper

(layer_regularization_losses
trainable_variables

)layers
regularization_losses
*metrics
	variables
+non_trainable_variables
I__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses"
_generic_user_object
,:*	
2dense-10-softmax_5/kernel
%:#
2dense-10-softmax_5/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper

,layer_regularization_losses
trainable_variables

-layers
regularization_losses
.metrics
	variables
/non_trainable_variables
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
'
00"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper

	1total
	2count
3
_fn_kwargs
4trainable_variables
5regularization_losses
6	variables
7	keras_api
N__call__
*O&call_and_return_all_conditional_losses"å
_tf_keras_layerË{"class_name": "MeanMetricWrapper", "name": "accuracy", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "accuracy", "dtype": "float32"}}
:  (2total
:  (2count
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
10
21"
trackable_list_wrapper

8layer_regularization_losses
4trainable_variables

9layers
5regularization_losses
:metrics
6	variables
;non_trainable_variables
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
0:.
2Adam/dense-128-relu_4/kernel/m
):'2Adam/dense-128-relu_4/bias/m
1:/	
2 Adam/dense-10-softmax_5/kernel/m
*:(
2Adam/dense-10-softmax_5/bias/m
0:.
2Adam/dense-128-relu_4/kernel/v
):'2Adam/dense-128-relu_4/bias/v
1:/	
2 Adam/dense-10-softmax_5/kernel/v
*:(
2Adam/dense-10-softmax_5/bias/v
2
.__inference_sequential_5_layer_call_fn_2097312
.__inference_sequential_5_layer_call_fn_2097408
.__inference_sequential_5_layer_call_fn_2097332
.__inference_sequential_5_layer_call_fn_2097399À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ì2é
"__inference__wrapped_model_2097215Â
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *2¢/
-*
flatten_5_inputÿÿÿÿÿÿÿÿÿ
ò2ï
I__inference_sequential_5_layer_call_and_return_conditional_losses_2097390
I__inference_sequential_5_layer_call_and_return_conditional_losses_2097280
I__inference_sequential_5_layer_call_and_return_conditional_losses_2097370
I__inference_sequential_5_layer_call_and_return_conditional_losses_2097291À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Õ2Ò
+__inference_flatten_5_layer_call_fn_2097419¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_flatten_5_layer_call_and_return_conditional_losses_2097414¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ú2×
0__inference_dense-128-relu_layer_call_fn_2097437¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
õ2ò
K__inference_dense-128-relu_layer_call_and_return_conditional_losses_2097430¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ü2Ù
2__inference_dense-10-softmax_layer_call_fn_2097455¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
÷2ô
M__inference_dense-10-softmax_layer_call_and_return_conditional_losses_2097448¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
<B:
%__inference_signature_wrapper_2097350flatten_5_input
Ì2ÉÆ
½²¹
FullArgSpec
args
jself
jinputs
varargs
 
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
Ì2ÉÆ
½²¹
FullArgSpec
args
jself
jinputs
varargs
 
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 °
"__inference__wrapped_model_2097215<¢9
2¢/
-*
flatten_5_inputÿÿÿÿÿÿÿÿÿ
ª "Cª@
>
dense-10-softmax*'
dense-10-softmaxÿÿÿÿÿÿÿÿÿ
®
M__inference_dense-10-softmax_layer_call_and_return_conditional_losses_2097448]0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 
2__inference_dense-10-softmax_layer_call_fn_2097455P0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ
­
K__inference_dense-128-relu_layer_call_and_return_conditional_losses_2097430^0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
0__inference_dense-128-relu_layer_call_fn_2097437Q0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ§
F__inference_flatten_5_layer_call_and_return_conditional_losses_2097414]3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
+__inference_flatten_5_layer_call_fn_2097419P3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿÀ
I__inference_sequential_5_layer_call_and_return_conditional_losses_2097280sD¢A
:¢7
-*
flatten_5_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 À
I__inference_sequential_5_layer_call_and_return_conditional_losses_2097291sD¢A
:¢7
-*
flatten_5_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 ·
I__inference_sequential_5_layer_call_and_return_conditional_losses_2097370j;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 ·
I__inference_sequential_5_layer_call_and_return_conditional_losses_2097390j;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 
.__inference_sequential_5_layer_call_fn_2097312fD¢A
:¢7
-*
flatten_5_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ

.__inference_sequential_5_layer_call_fn_2097332fD¢A
:¢7
-*
flatten_5_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ

.__inference_sequential_5_layer_call_fn_2097399];¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ

.__inference_sequential_5_layer_call_fn_2097408];¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
Æ
%__inference_signature_wrapper_2097350O¢L
¢ 
EªB
@
flatten_5_input-*
flatten_5_inputÿÿÿÿÿÿÿÿÿ"Cª@
>
dense-10-softmax*'
dense-10-softmaxÿÿÿÿÿÿÿÿÿ
