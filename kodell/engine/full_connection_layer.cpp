#include "stdafx.h"

#include <string.h>

#include "../engine/full_connection_layer.h"
#include "../engine/context_stack.h"
#include "../project/reporter_pool.h"
#include "../project/project_manager.h"
#include "../project/network_model.h"
#include "../util/util.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

CKDFullConnectionLayer::CKDFullConnectionLayer(const CKDFullLayerConfig* pLayerConfig, CKDDimension inputDim, CKDDimension outputDim) : CKDLayer()
{
	m_pLayerConfig = pLayerConfig;

	m_weight.SetSize(inputDim.getSize(), outputDim.getSize());
	m_weight.SetValue(DEF_PARAM_SMALL_POS);

	switch (m_pLayerConfig->m_bias) {
	case bias_each:
	case bias_on:
		m_bias.SetSize(outputDim.getSize());
		m_bias.SetValue(DEF_PARAM_SMALL_POS);
		break;
	case bias_common:
		m_bias.SetSize();
		m_bias.SetValue(DEF_PARAM_SMALL_POS);
		break;
	}

	if (m_pLayerConfig->m_inintVlaues.m_isvalid) {
		m_pLayerConfig->m_inintVlaues.weights.GetData(m_weight);
		m_pLayerConfig->m_inintVlaues.bias.GetData(m_bias);
	}
	else {
		m_weight.InitData(0, 0.01);
		m_bias.InitData(0, 0.01);
	}
}

CKDFullConnectionLayer::~CKDFullConnectionLayer()
{
}

CKDDimension CKDFullConnectionLayer::GetOutputDimension() const {
	return CKDDimension(1, m_weight.size(1));
}

void CKDFullConnectionLayer::ExecFowardStep(CKDContextStack& context, CKDReporterPool* pReporters) {
	m_execFowardStep(context, pReporters);
}

CKDTensor CKDFullConnectionLayer::m_execLenearTransform(CKDContextStack& context, CKDTensor input) {
	return m_weight.transpose_mult(input);
}

void CKDFullConnectionLayer::ExecBackwardStep(CKDContextStack& context, CKDReporterPool* pReporters) {
	m_execBackwardStep(context, pReporters);
}

void CKDFullConnectionLayer::ResetBackGradient() {
}

void CKDFullConnectionLayer::SaveFinalResult(CKDReporterPool* pReporters) {
	pReporters->ReportFinalResult(m_weight, m_bias);
}

double CKDFullConnectionLayer::GetWeightSqSum() const {
	return m_weight.l2_norm_sq();
}

double CKDFullConnectionLayer::GetWeightAbsSum() const {
	return m_weight.l1_norm();
}

void CKDFullConnectionLayer::m_evalWeightGradient(CKDContextStack& context, CKDTensor grad) {
	CKDTensor input = context.getInput();
	CKDTensor weight_delta = input.mult_cross(grad);

	context.accumulateLayerTensorXXX("weight_grad", weight_delta);
	context.accumulateLayerTensorXXX("weight_diff_sq_sum", weight_delta.l2_norm_sq());

	//pReporters->ReportLayerWeightGradDelta(weight_delta, weight_grad);
}

void CKDFullConnectionLayer::m_evalBiasGradient(CKDContextStack& context, CKDTensor grad) {
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

CKDTensor CKDFullConnectionLayer::m_evalInputGradient(CKDContextStack& context, CKDTensor grad) {
	CKDTensor delta_input = m_weight * grad;

	//pReporters->ReportLayerInputGrad(delta_input);

	return delta_input;
}

bool CKDFullConnectionLayer::FeedGradient(CKDContextStack& context, double rate, int batch_size, CKDReporterPool* pReporters, CKDProjectManager* pConfig) {
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

CKDOutputLayer::CKDOutputLayer(const CKDOutputLayerConfig* pLayerConfig, CKDDimension inputDim, CKDDimension outputDim)
: CKDFullConnectionLayer(pLayerConfig, inputDim, outputDim) {
	m_costFunc = pLayerConfig->m_costFunc;
}

CKDHiddenLayer::CKDHiddenLayer(const CKDHiddenLayerConfig* pLayerConfig, CKDDimension inputDim)
: CKDFullConnectionLayer(pLayerConfig, inputDim, CKDDimension(1, pLayerConfig->m_width)) {
}
