#include "stdafx.h"

#include <string.h>

#include "../engine/convolution_layer.h"
#include "../engine/context_stack.h"
#include "../project/reporter_pool.h"
#include "../project/project_manager.h"
#include "../project/network_model.h"
#include "../util/util.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

CKDConvolutionLayer::CKDConvolutionLayer(const CKDConvolutionLayerConfig* pLayerConfig, CKDDimension inputDim) : CKDLayer() {
	m_pLayerConfig = pLayerConfig;

	m_inputDimension = inputDim;
	m_outputDimension = inputDim.appendAxis(pLayerConfig->m_channels);

	CKDDimension weightDimension = m_outputDimension.clone();
	weightDimension.setAxisSize(0, pLayerConfig->m_width);
	weightDimension.setAxisSize(1, pLayerConfig->m_height);

	m_weight.SetDimension(weightDimension);
	m_weight.SetValue(DEF_PARAM_SMALL_POS);

	switch (m_pLayerConfig->m_bias) {
	case bias_each:
	case bias_on:
		m_bias.SetSize(pLayerConfig->m_channels);
		m_bias.SetValue(DEF_PARAM_SMALL_POS);
		break;
	case bias_common:
		m_bias.SetSize();
		m_bias.SetValue(DEF_PARAM_SMALL_POS);
		break;
	}

	m_weight.InitData(0, 0.01);
	m_bias.InitData(0, 0.01);

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
}

CKDConvolutionLayer::~CKDConvolutionLayer() {
}

CKDDimension CKDConvolutionLayer::GetOutputDimension() const {
	return m_outputDimension;
}

void CKDConvolutionLayer::ExecFowardStep(CKDContextStack& context, CKDReporterPool* pReporters) {
	m_execFowardStep(context, pReporters);
}

CKDTensor CKDConvolutionLayer::m_execLenearTransform(CKDContextStack& context, CKDTensor input) {
	return m_weight.convolution(input);
}

void CKDConvolutionLayer::m_evalWeightGradient(CKDContextStack& context, CKDTensor grad) {
	grad.ChangeDimension(m_outputDimension);

	CKDTensor input = context.getInput();
	CKDTensor weight_delta = input.inv_conv_weight(grad, m_weight.m_dimension); 

	context.accumulateLayerTensorXXX("weight_grad", weight_delta);
	context.accumulateLayerTensorXXX("weight_diff_sq_sum", weight_delta.l2_norm_sq());
}

void CKDConvolutionLayer::m_evalBiasGradient(CKDContextStack& context, CKDTensor grad) {
	CKDTensor diff;

	switch (m_pLayerConfig->m_bias) {
	case bias_each:
	case bias_on:
		diff = grad.channel_sum();
		context.accumulateLayerTensorXXX("bias_grad", diff);
		context.accumulateLayerTensorXXX("bias_diff_sq_sum", diff.l2_norm_sq());
		break;
	case bias_common:
		double bias_delta = grad.sum();
		context.accumulateLayerTensorXXX("bias_grad", bias_delta);
		context.accumulateLayerTensorXXX("bias_diff_sq_sum", bias_delta * bias_delta);
		break;
	}
}

CKDTensor CKDConvolutionLayer::m_evalInputGradient(CKDContextStack& context, CKDTensor grad) {
	CKDTensor delta_input = grad.inv_conv_input(m_weight);

	//pReporters->ReportLayerInputGrad(delta_input);

	return delta_input;
}

void CKDConvolutionLayer::ExecBackwardStep(CKDContextStack& context, CKDReporterPool* pReporters) {
	m_execBackwardStep(context, pReporters);
}

/*
void CKDConvolutionLayer::ExecFowardStep(CKDContextStack& context, CKDReporterPool* pReporters) {
	m_input = input;

	net = m_weight.convolution(input);

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
		break;
	case act_func_sigmoid:
		out = net.sigmoid();
		break;
	case act_func_tanh:
		out = net.tanh();
		break;
	case act_func_relu:
		out = net.relu();
		break;
	case act_func_softmax:
		out = net.softmax();
		break;
	default:
		THROW_TEMP;
	}

	if (bNeebCack) {
		switch (m_pLayerConfig->m_actFunc) {
		case act_func_sigmoid:
			m_jacobian = out.sigmoid_diff();
			//m_jacobian.ChangeDimension(CKDDimension(1, m_jacobian.size()));
			break;
		case act_func_tanh:
			m_jacobian = out.tanh_diff();
			//m_jacobian.ChangeDimension(CKDDimension(1, m_jacobian.size()));
			break;
		case act_func_relu:
			m_jacobian = out.relu_diff();
			//m_jacobian.ChangeDimension(CKDDimension(1, m_jacobian.size()));
			break;
		case act_func_softmax:
			m_jacobian = out.softmax_diff(output);
			break;
		}
	}

	if (pReporters) pReporters->ReportLayerActivateEval(m_pLayerConfig->m_actFunc, out);

	return out;
}
*/

/*
CKDTensor CKDConvolutionLayer::ExecFowardStep(bool bNeebCack, const CKDTensor input, const CKDTensor &output, CKDTensor &net, CKDReporterPool* pReporters) {
	m_input = input;

	net = m_weight.convolution(input);

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
		break;
	case act_func_sigmoid:
		out = net.sigmoid();
		break;
	case act_func_tanh:
		out = net.tanh();
		break;
	case act_func_relu:
		out = net.relu();
		break;
	case act_func_softmax:
		out = net.softmax();
		break;
	default:
		THROW_TEMP;
	}

	if (bNeebCack) {
		switch (m_pLayerConfig->m_actFunc) {
		case act_func_sigmoid:
			m_jacobian = out.sigmoid_diff();
			//m_jacobian.ChangeDimension(CKDDimension(1, m_jacobian.size()));
			break;
		case act_func_tanh:
			m_jacobian = out.tanh_diff();
			//m_jacobian.ChangeDimension(CKDDimension(1, m_jacobian.size()));
			break;
		case act_func_relu:
			m_jacobian = out.relu_diff();
			//m_jacobian.ChangeDimension(CKDDimension(1, m_jacobian.size()));
			break;
		case act_func_softmax:
			m_jacobian = out.softmax_diff(output);
			break;
		}
	}

	if (pReporters) pReporters->ReportLayerActivateEval(m_pLayerConfig->m_actFunc, out);

	return out;
}

CKDTensor CKDConvolutionLayer::ExecBackwardStep(const CKDTensor gradient, CKDReporterPool* pReporters) {
	//grad = grad.mult_each(m_grad);
	CKDTensor grad = gradient;
	if (m_pLayerConfig->m_actFunc != act_func_bypass) {
		grad = m_jacobian.mult_each(gradient);
	}

	grad.ChangeDimension(m_outputDimension);

	pReporters->ReportLayerGradient(grad);

	//CKDMatrix weight_grad = grad.mult_cross(m_input);

	CKDTensor m_weight_delta = m_input.inv_conv_weight(grad, m_weight.m_dimension);
	pReporters->ReportLayerWeightGradDelta(m_weight_delta, m_weight_grad);
	m_weight_grad += m_weight_delta;
	m_weight_diff_sq_sum += m_weight_delta.l2_norm_sq();

	pReporters->ReportLayerWeightGradDelta(m_weight_delta, m_weight_grad);

	CKDTensor diff;

	switch (m_pLayerConfig->m_bias) {
	case bias_each:
	case bias_on:
		diff = grad.channel_sum();
		m_bias_grad += diff;
		pReporters->ReportLayerBiasGradDelta(diff, m_bias_grad);
		m_bias_diff_sq_sum += diff.l2_norm_sq();
		break;
	case bias_common:
		double bias_delta = grad.sum();
		m_bias_diff += bias_delta;
		pReporters->ReportLayerBiasGradDelta(bias_delta, m_bias_diff);
		m_bias_diff_sq_sum += bias_delta * bias_delta;
		break;
	}

	CKDTensor delta_input = grad.inv_conv_input(m_weight);

	pReporters->ReportLayerInputGrad(delta_input);

	return delta_input;
}
*/

void CKDConvolutionLayer::ResetBackGradient() {
	/*
	m_weight_grad.Reset();
	m_bias_grad.Reset();
	m_bias_diff = 0;
	m_weight_diff_sq_sum = 0;
	m_bias_diff_sq_sum = 0;
	*/
}

bool CKDConvolutionLayer::FeedGradient(CKDContextStack& context, double rate, int batch_size, CKDReporterPool* pReporters, CKDProjectManager* pConfig) {
	CKDTensor weight_grad = context.getLayerTensorXXX("weight_grad");

	CKDTensor diff = weight_grad / batch_size;

	diff += weight_grad * pConfig->GetWeightDecayNorm2Ratio();

	m_weight = m_weight - diff * rate;

	double weight_diff_sq_sum = context.getLayerScalarXXX("weight_diff_sq_sum");
	double bias_diff_sq_sum = 0;

	pReporters->ReportLayerFeedSetting(batch_size, rate);
	pReporters->ReportLayerFeedWeight(weight_grad, m_weight);

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

void CKDConvolutionLayer::SaveFinalResult(CKDReporterPool* pReporters) {
	pReporters->ReportFinalResult(m_weight, m_bias);
}

double CKDConvolutionLayer::GetWeightSqSum() const {
	return m_weight.l2_norm_sq();
}

double CKDConvolutionLayer::GetWeightAbsSum() const {
	return m_weight.l1_norm();
}
