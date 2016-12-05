#include "stdafx.h"

#include <string.h>

#include "../engine/recurrent_layer.h"
#include "../engine/context_stack.h"
#include "../project/reporter_pool.h"
#include "../project/project_manager.h"
#include "../project/network_model.h"
#include "../util/util.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

CKDRecurrentLayer::CKDRecurrentLayer(const CKDRecurrentLayerConfig* pLayerConfig, CKDDimension inputDim) : CKDLayer() {
	m_pLayerConfig = pLayerConfig;

	int width = m_pLayerConfig->m_width;

	m_inputWeight.SetSize(inputDim.getSize(), width);
	m_inputWeight.SetValue(DEF_PARAM_SMALL_POS);

	m_recurWeight.SetSize(width, width);
	m_recurWeight.SetValue(DEF_PARAM_SMALL_POS);

	switch (m_pLayerConfig->m_bias) {
	case bias_each:
	case bias_on:
		m_bias.SetSize(width);
		m_bias.SetValue(DEF_PARAM_SMALL_POS);
		break;
	case bias_common:
		m_bias.SetSize();
		m_bias.SetValue(DEF_PARAM_SMALL_POS);
		break;
	}

	/*
	if (m_pLayerConfig->m_inintVlaues.m_isvalid) {
		m_pLayerConfig->m_inintVlaues.weights.GetData(m_weight);
		m_pLayerConfig->m_inintVlaues.bias.GetData(m_bias);
	}
	else {
		m_weight.InitData(0, 0.01);
		m_bias.InitData(0, 0.01);
	}
	*/


	m_inputDimension = inputDim;
	m_outputDimension = CKDDimension(1, width);
}

CKDRecurrentLayer::~CKDRecurrentLayer() {
}

CKDDimension CKDRecurrentLayer::GetOutputDimension() const {
	return m_outputDimension;
}

void CKDRecurrentLayer::ExecFowardStep(CKDContextStack& context, CKDReporterPool* pReporters) {
	CKDTensor input = context.getInput();

	CKDTensor linear = m_execLenearTransform(context, input);
	CKDTensor affine = m_execAddBias(context, linear);
	CKDTensor result = m_execActivateFunc(context, affine);

	//context.setLayerInfo("output", result);
	context.setOutput(result);
}

CKDTensor CKDRecurrentLayer::m_execLenearTransform(CKDContextStack& context, CKDTensor input) {
	CKDTensor recur = context.getOutput(-1);

	if (recur.is_empty()) recur.SetDimension(m_outputDimension);

	CKDTensor net1 = m_inputWeight.transpose_mult(input);
	CKDTensor net2 = m_recurWeight.transpose_mult(recur);

	return net1 + net2;
}

void CKDRecurrentLayer::ExecBackwardStep(CKDContextStack& context, CKDReporterPool* pReporters) {
	CKDTensor gradient1 = context.getGradient();
	CKDTensor gradient2 = context.getInstantTensor("recur_grad", 1);

	if (gradient2.is_empty()) gradient2.SetDimension(gradient1.m_dimension);

	CKDTensor gradient = gradient1 + gradient2;

	CKDTensor out = context.getOutput();
	CKDTensor grad = m_execDeactivateFunc(context, gradient, out);

	m_evalInputWeightGradient(context, grad);
	m_evalRecurWeightGradient(context, grad);
	m_evalBiasGradient(context, grad);

	if (context.get_need_input_grad()) {
		context.setGradient(m_evalInputGradient(context, grad));
	}

	if (context.get_need_recur_grad()) {
		context.setInstantTensor("recur_grad", m_evalRecurGradient(context, grad));
	}
}

void CKDRecurrentLayer::m_evalInputWeightGradient(CKDContextStack& context, CKDTensor grad) {
	CKDTensor input = context.getInput();
	CKDTensor weight_delta = input.mult_cross(grad);

	context.accumulateLayerTensorXXX("input_weight_grad", weight_delta);
	context.accumulateLayerTensorXXX("weight_diff_sq_sum", weight_delta.l2_norm_sq());

	//pReporters->ReportLayerWeightGradDelta(weight_delta, weight_grad);
}

void CKDRecurrentLayer::m_evalRecurWeightGradient(CKDContextStack& context, CKDTensor grad) {
	CKDTensor output = context.getOutput(-1);

	if (output.is_empty()) return;

	CKDTensor weight_delta = output.mult_cross(grad);

	context.accumulateLayerTensorXXX("recur_weight_grad", weight_delta);
	context.accumulateLayerTensorXXX("weight_diff_sq_sum", weight_delta.l2_norm_sq());

	//pReporters->ReportLayerWeightGradDelta(weight_delta, weight_grad);
}

void CKDRecurrentLayer::m_evalBiasGradient(CKDContextStack& context, CKDTensor grad) {
	switch (m_pLayerConfig->m_bias) {
	case bias_each:
	case bias_on:
		context.accumulateLayerTensorXXX("bias_grad", grad);
		context.accumulateLayerTensorXXX("bias_diff_sq_sum", grad.l2_norm_sq());
		//pReporters->ReportLayerBiasGradDelta(grad, m_bias_grad);
		break;
	case bias_common:
		double bias_delta = grad.sum();
		context.accumulateLayerTensorXXX("bias_grad", bias_delta);
		context.accumulateLayerTensorXXX("bias_diff_sq_sum", bias_delta * bias_delta);
		//pReporters->ReportLayerBiasGradDelta(bias_delta, m_bias_diff);
		break;
	}
}

CKDTensor CKDRecurrentLayer::m_evalRecurGradient(CKDContextStack& context, CKDTensor grad) {
	CKDTensor delta_recur = m_recurWeight * grad;

	//pReporters->ReportLayerInputGrad(delta_input);

	return delta_recur;
}

/*
CKDTensor CKDRecurrentLayer::ExecFowardStep(bool bNeebCack, const CKDTensor input, const CKDTensor &output, CKDTensor &net, CKDReporterPool* pReporters) {
	if (m_output.is_empty()) {
		m_output = CKDTensor(m_outputDimension);
	}

	m_input = input;

	CKDTensor input_ex = input.concat_to_vector(m_output);

	net = m_weight.transpose_mult(input_ex);

	switch (m_pLayerConfig->m_bias) {
	case bias_each:
	case bias_on:
		net = net + m_bias;
		break;
	case bias_common:
		net = net + m_bias[0];
		break;
	}

	if (pReporters) pReporters->ReportLayerLinearEval(m_weight, m_bias, net);

	CKDTensor out;

	switch (m_pLayerConfig->m_actFunc) {
	case act_func_bypass:
		out = net;
		//if (bNeebCack) m_jacobian.SetScalar(1); //out.fill(1).to_diagonal();
		break;
	case act_func_sigmoid:
		out = net.sigmoid();
		if (bNeebCack) m_jacobian = out.sigmoid_diff();
		break;
	case act_func_tanh:
		out = net.tanh();
		if (bNeebCack) m_jacobian = out.tanh_diff();
		break;
	case act_func_relu:
		out = net.relu();
		if (bNeebCack) m_jacobian = out.relu_diff();
		break;
	case act_func_softmax:
		out = net.softmax();
		if (bNeebCack) m_jacobian = out.softmax_diff(output);
		break;
	default:
		THROW_TEMP;
	}

	if (pReporters) pReporters->ReportLayerActivateEval(m_pLayerConfig->m_actFunc, out);

	m_output = out;

	return out;
}

CKDTensor CKDRecurrentLayer::ExecBackwardStep(const CKDTensor gradient, CKDReporterPool* pReporters) {
	THROW_TEMP;
	return CKDTensor();
}
*/

void CKDRecurrentLayer::ResetBackGradient() {
	/*
	m_weight_grad.Reset();
	m_bias_grad.Reset();
	m_bias_diff = 0;
	m_weight_diff_sq_sum = 0;
	m_bias_diff_sq_sum = 0;
	*/
}

bool CKDRecurrentLayer::FeedGradient(CKDContextStack& context, double rate, int batch_size, CKDReporterPool* pReporters, CKDProjectManager* pConfig) {
	CKDTensor input_weight_grad = context.getLayerTensorXXX("input_weight_grad");
	CKDTensor recur_weight_grad = context.getLayerTensorXXX("recur_weight_grad");

	CKDTensor input_diff = input_weight_grad / batch_size;
	CKDTensor recur_diff = recur_weight_grad / batch_size;

	input_diff += input_weight_grad * pConfig->GetWeightDecayNorm2Ratio();
	recur_diff += recur_weight_grad * pConfig->GetWeightDecayNorm2Ratio();

	m_inputWeight = m_inputWeight - input_diff * rate;
	m_recurWeight = m_recurWeight - recur_diff * rate;

	double weight_diff_sq_sum = context.getLayerScalarXXX("weight_diff_sq_sum");
	double bias_diff_sq_sum = 0;

	pReporters->ReportLayerFeedSetting(batch_size, rate);
	//pReporters->ReportLayerFeedWeight(weight_grad, m_weight);

	CKDTensor bias_grad;

	switch (m_pLayerConfig->m_bias) {
	case bias_each:
	case bias_on:
		bias_grad = context.getLayerTensorXXX("bias_grad");
		bias_diff_sq_sum = context.getLayerScalarXXX("bias_diff_sq_sum");
		m_bias = m_bias - bias_grad / batch_size * rate;
		pReporters->ReportLayerFeedBias(bias_grad, m_bias);
		break;
	case bias_common:
		bias_grad = context.getLayerTensorXXX("bias_grad");
		bias_diff_sq_sum = context.getLayerScalarXXX("bias_diff_sq_sum");
		m_bias = m_bias - bias_grad / batch_size * rate;
		pReporters->ReportLayerFeedBias(bias_grad, m_bias);
		break;
	}

	return near_zero(weight_diff_sq_sum) && near_zero(bias_diff_sq_sum);
}

void CKDRecurrentLayer::SaveFinalResult(CKDReporterPool* pReporters) {
	pReporters->ReportFinalResult(m_inputWeight, m_recurWeight, m_bias);
}

double CKDRecurrentLayer::GetWeightSqSum() const {
	return m_inputWeight.l2_norm_sq() + m_recurWeight.l2_norm_sq();
}

double CKDRecurrentLayer::GetWeightAbsSum() const {
	return m_inputWeight.l1_norm() + m_recurWeight.l1_norm();
}
