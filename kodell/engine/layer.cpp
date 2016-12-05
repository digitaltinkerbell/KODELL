#include "stdafx.h"

#include <string.h>

#include "../engine/layer.h"
#include "../engine/full_connection_layer.h"
#include "../engine/convolution_layer.h"
#include "../engine/pooling_layer.h"
#include "../engine/recurrent_layer.h"
#include "../engine/context_stack.h"
#include "../project/reporter_pool.h"
#include "../project/project_manager.h"
#include "../project/network_model.h"
#include "../util/util.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

CKDLayer* CKDLayer::Create(const CKDLayerConfig* pLayerConfig, CKDDimension inputDim, CKDDimension outputDim) {
	CKDLayer* pLayer = NULL;

	switch (pLayerConfig->m_layerType) {
	case layer_output:
		pLayer = new CKDOutputLayer((const CKDOutputLayerConfig*)pLayerConfig, inputDim, outputDim);
		break;
	case layer_full:
		pLayer = new CKDHiddenLayer((const CKDHiddenLayerConfig*)pLayerConfig, inputDim);
		break;
	case layer_convolution:
		pLayer = new CKDConvolutionLayer((const CKDConvolutionLayerConfig*)pLayerConfig, inputDim);
		break;
	case layer_pooling:
		pLayer = new CKDPoolingLayer((const CKDPoolingLayerConfig*)pLayerConfig, inputDim);
		break;
	case layer_recurrent:
		pLayer = new CKDRecurrentLayer((const CKDRecurrentLayerConfig*)pLayerConfig, inputDim);
		break;
	}

	return pLayer;
}

CKDLayer::CKDLayer()
{
}

CKDLayer::~CKDLayer()
{
}

void CKDLayer::m_execFowardStep(CKDContextStack& context, CKDReporterPool* pReporters) {
	CKDTensor input = context.getInput();

	CKDTensor linear = m_execLenearTransform(context, input);
	CKDTensor affine = m_execAddBias(context, linear);
	CKDTensor result = m_execActivateFunc(context, affine);

	//context.setLayerInfo("output", result);
	context.setOutput(result);
}

CKDTensor CKDLayer::m_execLenearTransform(CKDContextStack& context, CKDTensor input) {
	THROW_TEMP;
}

CKDTensor CKDLayer::m_execAddBias(CKDContextStack& context, CKDTensor linear) {
	switch (m_get_bias_mode()) {
	case bias_each:
	case bias_on:
		return linear + m_get_bias_vector();
	case bias_common:
		return linear + m_get_bias_value();
	default:
		return linear;
	}
}

CKDTensor CKDLayer::m_execActivateFunc(CKDContextStack& context, CKDTensor affine) {
	switch (m_get_activate_func()) {
	case act_func_bypass:
		return affine;
	case act_func_sigmoid:
		return affine.sigmoid();
	case act_func_tanh:
		return affine.tanh();
	case act_func_relu:
		return affine.relu();
	case act_func_softmax:
		return affine.softmax();
	default:
		THROW_TEMP;
	}
}

void CKDLayer::m_execBackwardStep(CKDContextStack& context, CKDReporterPool* pReporters) {
	CKDTensor gradient = context.getGradient();
	CKDTensor out = context.getOutput();
	CKDTensor grad = m_execDeactivateFunc(context, gradient, out);

	m_evalWeightGradient(context, grad);
	m_evalBiasGradient(context, grad);
	
	if (context.get_need_input_grad()) {
		context.setGradient(m_evalInputGradient(context, grad));
	}
}

CKDTensor CKDLayer::m_execDeactivateFunc(CKDContextStack& context, CKDTensor gradient, CKDTensor out) {
	switch (m_get_activate_func()) {
	case act_func_softmax:
		{
			 CKDTensor jacobian = out.softmax_diff(context.getOutput());
			 return jacobian * gradient;
		}
	case act_func_sigmoid:
		return gradient.mult_each(out.sigmoid_diff());
	case act_func_tanh:
		return gradient.mult_each(out.tanh_diff());
	case act_func_relu:
		return gradient.mult_each(out.relu_diff());
	case act_func_bypass:
		return gradient;
	default:
		THROW_TEMP;
	}
}

	/*
	CKDTensor jacobian;

	switch (m_pLayerConfig->m_actFunc) {
	case act_func_softmax:
		jacobian = out.softmax_diff(context.getExampleAnswer());
		grad = jacobian * gradient;
		break;
	case act_func_sigmoid:
		grad = gradient.mult_each(out.sigmoid_diff());
		break;
	case act_func_tanh:
		grad = gradient.mult_each(out.tanh_diff());
		break;
	case act_func_relu:
		grad = gradient.mult_each(out.relu_diff());
		break;
	case act_func_bypass:
		grad = gradient;
		break;
	default:
		THROW_TEMP;
	}

	pReporters->ReportLayerGradient(grad);
	*/

/*
	CKDTensor m_weight_delta = m_input.mult_cross(grad);
	pReporters->ReportLayerWeightGradDelta(m_weight_delta, m_weight_grad);
	m_weight_grad += m_weight_delta;
	m_weight_diff_sq_sum += m_weight_delta.l2_norm_sq();

	pReporters->ReportLayerWeightGradDelta(m_weight_delta, m_weight_grad);

	switch (m_pLayerConfig->m_bias) {
	case bias_each:
	case bias_on:
		m_bias_grad += grad;
		pReporters->ReportLayerBiasGradDelta(grad, m_bias_grad);
		m_bias_diff_sq_sum += grad.l2_norm_sq();
		break;
	case bias_common:
		double bias_delta = grad.sum();
		m_bias_diff += bias_delta;
		pReporters->ReportLayerBiasGradDelta(bias_delta, m_bias_diff);
		m_bias_diff_sq_sum += bias_delta * bias_delta;
		break;
	}

	CKDTensor delta_input = m_weight * grad;

	pReporters->ReportLayerInputGrad(delta_input);

	context.pushCurrentGradient(delta_input);
}
*/

	/*
	//if (pReporters) pReporters->ReportLayerLinearEval(m_weight, m_bias, affine);

	CKDTensor linear = m_weight.transpose_mult(input);
	CKDTensor affine;
	CKDTensor out;

	switch (m_pLayerConfig->m_bias) {
	case bias_each:
	case bias_on:
		affine = linear + m_bias;
		break;
	case bias_common:
		affine = linear + m_bias[0];
		break;
	default:
		affine = linear;
		break;
	}

	if (pReporters) pReporters->ReportLayerLinearEval(m_weight, m_bias, affine);

	switch (m_pLayerConfig->m_actFunc) {
	case act_func_bypass:
		out = affine;
		//if (bNeebCack) m_jacobian.SetScalar(1); //out.fill(1).to_diagonal();
		break;
	case act_func_sigmoid:
		out = affine.sigmoid();
		//if (bNeedBack) m_jacobian = out.sigmoid_diff();
		break;
	case act_func_tanh:
		out = affine.tanh();
		//if (bNeedBack) m_jacobian = out.tanh_diff();
		break;
	case act_func_relu:
		out = affine.relu();
		//if (bNeedBack) m_jacobian = out.relu_diff();
		break;
	case act_func_softmax:
		out = affine.softmax();
		//if (bNeedBack) m_jacobian = out.softmax_diff(output);
		break;
	default:
		THROW_TEMP;
	}

	if (pReporters) pReporters->ReportLayerActivateEval(m_pLayerConfig->m_actFunc, out);

	context.setLayerInfo("output", out);
	context.pushCurrentResult(out);
	*/
