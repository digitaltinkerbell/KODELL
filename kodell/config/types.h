#pragma once

#ifndef _LINUX_
#include <direct.h>
#define PATH_DELIMETER '\\'
#else
#include <unistd.h>
#define PATH_DELIMETER '/'
#endif

#include "../config/tinyxml2.h"

using namespace tinyxml2;

enum enumProjectMode { proj_mode_undef = -1, proj_mode_regression = 0, proj_mode_binary = 1, proj_mode_classify = 2, };
enum enumProjectType { proj_type_undef = -1, proj_type_normal = 0, proj_type_recurrent = 1, };

enum enumTrainMethod {
	train_undef=-1, train_sg = 0, train_sgd = 1, train_sgdnm = 2, train_rmsprop = 3, train_rmsprop_nm = 4,
	train_adagrad = 5, train_adadelta = 6, train_adam = 7, train_newton = 8,
};
enum enumReportType { report_undef=-1, report_full=0, report_brief=1, };

enum enumMinibatchType { minibatch_undef = -1, minibatch_random = 0, minibatch_sequential = 1, minibatch_all = 2, minibatch_unique = 3, };

enum enumTestDest { test_dest_train, test_dest_testset, test_dest_trainset, };

enum enumRegularType { regular_type_undef = -1, regular_type_norm1 = 0, regular_type_norm2 = 1, };
enum enumRegulatrDest { regular_dest_undef = -1, regular_dest_all = 0, };

enum enumLayerType { layer_undef = -1, layer_output = 0, layer_full = 1, layer_convolution = 2, layer_pooling = 3, layer_recurrent = 4, };
enum costFuncType { cost_func_undef = -1, cost_func_mse = 0, cost_func_cross_entropy = 1, };
enum actFuncType { act_func_undef = -1, act_func_bypass = 0, act_func_sigmoid = 1, act_func_tanh = 2, act_func_relu = 3, act_func_softmax = 4, };
enum poolingFuncType { pooling_func_undef = -1, pooling_func_max = 0 };
enum biasType { bias_undef = -1, bias_off = 0, bias_on = 1, bias_each = 2, bias_common = 3, };

enum enumTrainPhase { train_phase_forward, train_phase_backward, train_phase_feed, train_phase_final, };
enum enumReportDest { enum_report_dest_none, enum_report_dest_file, enum_report_dest_stdout, enum_report_dest_callback, };

