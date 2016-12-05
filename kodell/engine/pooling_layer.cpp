#include "stdafx.h"

#include <string.h>

#include "../engine/pooling_layer.h"
#include "../engine/context_stack.h"
#include "../project/reporter_pool.h"
#include "../project/project_manager.h"
#include "../project/network_model.h"
#include "../util/util.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

CKDPoolingLayer::CKDPoolingLayer(const CKDPoolingLayerConfig* pLayerConfig, CKDDimension inputDim) : CKDLayer() {
	m_pLayerConfig = pLayerConfig;

	m_inputDimension = inputDim;
	m_outputDimension = inputDim.clone();
	
	int sx = m_pLayerConfig->m_stride_x;
	int sy = m_pLayerConfig->m_stride_y;

	m_outputDimension.setAxisSize(0, (inputDim.getAxisSize(0) + sx - 1) / sx);
	m_outputDimension.setAxisSize(1, (inputDim.getAxisSize(1) + sy - 1) / sy);

	m_router = new int[m_outputDimension.getSize()];
}

CKDPoolingLayer::~CKDPoolingLayer() {
	delete[] m_router;
}

CKDDimension CKDPoolingLayer::GetOutputDimension() const {
	return m_outputDimension;
}

void CKDPoolingLayer::ExecFowardStep(CKDContextStack& context, CKDReporterPool* pReporters) {
	CKDTensor input = context.getInput();

	CKDTensor submap = m_execPoolingFunc(context, input);
	CKDTensor result = m_execActivateFunc(context, submap);

	context.setOutput(result);
}

CKDTensor CKDPoolingLayer::m_execPoolingFunc(CKDContextStack& context, CKDTensor input) {
	int sx = m_pLayerConfig->m_stride_x;
	int sy = m_pLayerConfig->m_stride_y;

	switch (m_pLayerConfig->m_poolFunc) {
	case pooling_func_max:
		return input.pooling_max(sx, sy, m_router);
	}

	THROW_TEMP;
}

void CKDPoolingLayer::ExecBackwardStep(CKDContextStack& context, CKDReporterPool* pReporters) {
	CKDTensor gradient = context.getGradient();
	CKDTensor out = context.getOutput();
	CKDTensor grad = m_execDeactivateFunc(context, gradient, out);

	/*
	m_evalWeightGradient(context, grad);
	m_evalBiasGradient(context, grad);

	if (context.get_need_input_grad()) {
		context.pushCurrentGradient(m_evalInputGradient(context, grad));
	}

	THROW_TEMP;
	*/

	/*
	CKDTensor grad = gradient;
	if (m_pLayerConfig->m_actFunc != act_func_bypass) {
		grad = m_jacobian.mult_each(gradient);
	}
	*/


	//pReporters->ReportLayerGradient(grad);

	//CKDTensor delta_input(m_inputDimension);

	//delta_input.route_gradient(grad, m_router);

	//pReporters->ReportLayerInputGrad(delta_input);

	//return delta_input;

	if (context.get_need_input_grad()) {
		context.setGradient(m_evalInputGradient(context, grad));
	}
}

CKDTensor CKDPoolingLayer::m_evalInputGradient(CKDContextStack& context, CKDTensor grad) {
	CKDTensor delta_input(m_inputDimension);

	grad.ChangeDimension(m_outputDimension);
	delta_input.route_gradient(grad, m_router);

	return delta_input;
}

/*
CKDTensor CKDPoolingLayer::ExecFowardStep(bool bNeebCack, const CKDTensor input, const CKDTensor &output, CKDTensor &net, CKDReporterPool* pReporters) {
	int sx = m_pLayerConfig->m_stride_x;
	int sy = m_pLayerConfig->m_stride_y;

	//net = m_weight.convolution(input);

	switch (m_pLayerConfig->m_poolFunc) {
	case pooling_func_max:
		net = input.pooling_max(sx, sy, m_router);
		break;
	}

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

CKDTensor CKDPoolingLayer::ExecBackwardStep(const CKDTensor gradient, CKDReporterPool* pReporters) {
	CKDTensor grad = gradient;
	if (m_pLayerConfig->m_actFunc != act_func_bypass) {
		grad = m_jacobian.mult_each(gradient);
	}

	grad.ChangeDimension(m_outputDimension);

	pReporters->ReportLayerGradient(grad);

	CKDTensor delta_input(m_inputDimension);
	
	delta_input.route_gradient(grad, m_router);

	pReporters->ReportLayerInputGrad(delta_input);

	return delta_input;
}
*/

void CKDPoolingLayer::ResetBackGradient() {
}

bool CKDPoolingLayer::FeedGradient(CKDContextStack& context, double rate, int batch_size, CKDReporterPool* pReporters, CKDProjectManager* pConfig) {
	return true;
}

void CKDPoolingLayer::SaveFinalResult(CKDReporterPool* pReporters) {
}

double CKDPoolingLayer::GetWeightSqSum() const {
	return 0;
}

double CKDPoolingLayer::GetWeightAbsSum() const {
	return 0;
}
