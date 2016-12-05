#include "stdafx.h"

#include "../config/types.h"
#include "../project/reporter_pool.h"
#include "../util/xml_util.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

CKDReporterPool::CKDReporterPool() {
	m_nReporterCount = 0;
	m_nReportFlags = 0;
	m_nIterateStep = -1;
	m_nLayer = -1;

	m_pReporters = NULL;
}

CKDReporterPool::~CKDReporterPool() {
	delete[] m_pReporters;
}

void CKDReporterPool::Setup(XMLNode* node) {
	int nCount = CKDXmlUtil::GetTagChildCount(node, "report");

	XMLNode* child = CKDXmlUtil::SeekTagChild(node, "report");

	m_nReporterCount = 0;
	m_nReportFlags = 0;

	m_pReporters = new CKDReporter[nCount];

	for (int i = 0; i < nCount; i++) {
		m_pReporters[m_nReporterCount].Setup(child);
		m_nReportFlags |= m_pReporters[m_nReporterCount].m_nReportFlags;
		
		m_nReporterCount++;

		child = CKDXmlUtil::SeekTagSibling(child, "report");
	}
}

void CKDReporterPool::ReportDebug(const char* caption, const CKDTensor& tensor) {
	if (!m_flagCheck(REPORT_FLAG_TRAIN_START)) return;

	for (int i = 0; i < m_nReporterCount; i++) {
		m_pReporters[i].ReportDebug(caption, tensor);
	}
}

void CKDReporterPool::ReportTrainStart(const char* pProjectName) {
	if (!m_flagCheck(REPORT_FLAG_TRAIN_START)) return;

	for (int i = 0; i < m_nReporterCount; i++) {
		m_pReporters[i].ReportTrainStart(pProjectName);
	}
}

void CKDReporterPool::ReportTrainExit(const char* pReason) {
	if (!m_flagCheck(REPORT_FLAG_TRAIN_EXIT)) return;

	for (int i = 0; i < m_nReporterCount; i++) {
		m_pReporters[i].ReportTrainExit(pReason);
	}
}

void CKDReporterPool::ReportTrainEnd(int nStep) {
	if (!m_flagCheck(REPORT_FLAG_TRAIN_END)) return;

	for (int i = 0; i < m_nReporterCount; i++) {
		m_pReporters[i].ReportTrainEnd(nStep);
	}
}

void CKDReporterPool::ReportIterateStepStart(int nStep) {
	if (!m_flagCheck(REPORT_FLAG_STEP_START)) return;

	m_nIterateStep = nStep;

	for (int i = 0; i < m_nReporterCount; i++) {
		m_pReporters[i].ReportIterateStepStart(nStep);
	}
}

void CKDReporterPool::ReportIterateStepEnd() {
	if (!m_flagCheck(REPORT_FLAG_STEP_END)) return;

	m_nIterateStep = -1;

	for (int i = 0; i < m_nReporterCount; i++) {
		m_pReporters[i].ReportIterateStepEnd(m_nIterateStep);
	}
}

void CKDReporterPool::ReportMinibatchSelect(int nth, int src_idx) {
	if (!m_flagCheck(REPORT_FLAG_MINIBATCH_SELECT)) return;

	for (int i = 0; i < m_nReporterCount; i++) {
		m_pReporters[i].ReportMinibatchSelect(m_nIterateStep, nth, src_idx);
	}
}

void CKDReporterPool::ReportMinibatchVomit(int nth) {
	if (!m_flagCheck(REPORT_FLAG_MINIBATCH_VOMIT)) return;

	for (int i = 0; i < m_nReporterCount; i++) {
		m_pReporters[i].ReportMinibatchVomit(m_nIterateStep, nth);
	}
}

void CKDReporterPool::ReportExample(CKDTensor input, CKDTensor output) {
	if (!m_flagCheck(REPORT_FLAG_EXAMPLE)) return;

	for (int i = 0; i < m_nReporterCount; i++) {
		m_pReporters[i].ReportExample(m_nIterateStep, input, output);
	}
}

void CKDReporterPool::ReportSetLayer(int nLayer, int nLayerCount, enum enumTrainPhase trainPhase) {
	if (!m_flagCheck(REPORT_FLAG_SET_LAYER)) return;

	m_nLayer = nLayer;

	for (int i = 0; i < m_nReporterCount; i++) {
		m_pReporters[i].ReportSetLayer(m_nIterateStep, m_nLayer, nLayerCount, trainPhase);
	}
}

void CKDReporterPool::ReportLayerLinearEval(CKDTensor weight, CKDTensor bias, CKDTensor net) {
	if (!m_flagCheck(REPORT_FLAG_LAYER_LINEAR_EVAL)) return;

	for (int i = 0; i < m_nReporterCount; i++) {
		m_pReporters[i].ReportLayerLinearEval(m_nIterateStep, m_nLayer, weight, bias, net);
	}
}

void CKDReporterPool::ReportLayerActivateEval(enum actFuncType func, CKDTensor out) {
	if (!m_flagCheck(REPORT_FLAG_LAYER_ACTIVATE_EVAL)) return;

	for (int i = 0; i < m_nReporterCount; i++) {
		m_pReporters[i].ReportLayerActivateEval(m_nIterateStep, m_nLayer, func, out);
	}
}

void CKDReporterPool::ReportCost(double cost, bool reset, bool close) {
	if (!m_flagCheck(REPORT_FLAG_COST) && !m_flagCheck(REPORT_FLAG_COST_AVG)) return;

	for (int i = 0; i < m_nReporterCount; i++) {
		m_pReporters[i].ReportCost(m_nIterateStep, cost, reset, close);
	}
}

void CKDReporterPool::ReportCostGradient(CKDTensor grad) {
	if (!m_flagCheck(REPORT_FLAG_COST_GRAD)) return;

	for (int i = 0; i < m_nReporterCount; i++) {
		m_pReporters[i].ReportCostGradient(m_nIterateStep, grad);
	}
}

void CKDReporterPool::ReportLayerGradient(CKDTensor grad) {
	if (!m_flagCheck(REPORT_FLAG_LAYER_GRAD)) return;

	for (int i = 0; i < m_nReporterCount; i++) {
		m_pReporters[i].ReportLayerGradient(m_nIterateStep, m_nLayer, grad);
	}
}

void CKDReporterPool::ReportLayerWeightGradDelta(CKDTensor delta, CKDTensor acc) {
	if (!m_flagCheck(REPORT_FLAG_LAYER_GRAD_DELTA)) return;

	for (int i = 0; i < m_nReporterCount; i++) {
		m_pReporters[i].ReportLayerWeightGradDelta(m_nIterateStep, m_nLayer, delta, acc);
	}
}

void CKDReporterPool::ReportLayerBiasGradDelta(CKDTensor delta, CKDTensor acc) {
	if (!m_flagCheck(REPORT_FLAG_LAYER_BIAS_DELTA)) return;

	for (int i = 0; i < m_nReporterCount; i++) {
		m_pReporters[i].ReportLayerBiasGradDelta(m_nIterateStep, m_nLayer, delta, acc);
	}
}

void CKDReporterPool::ReportLayerBiasGradDelta(double delta, double acc) {
	if (!m_flagCheck(REPORT_FLAG_LAYER_BIAS_DELTA)) return;

	for (int i = 0; i < m_nReporterCount; i++) {
		m_pReporters[i].ReportLayerBiasGradDelta(m_nIterateStep, m_nLayer, delta, acc);
	}
}

void CKDReporterPool::ReportLayerInputGrad(CKDTensor delta_input) {
	if (!m_flagCheck(REPORT_FLAG_LAYER_INPUT_GRAD)) return;

	for (int i = 0; i < m_nReporterCount; i++) {
		m_pReporters[i].ReportLayerInputGrad(m_nIterateStep, m_nLayer, delta_input);
	}
}

void CKDReporterPool::ReportLayerFeedSetting(int batch_size, double rate) {
	if (!m_flagCheck(REPORT_FLAG_LAYER_FEED_SETTING)) return;

	for (int i = 0; i < m_nReporterCount; i++) {
		m_pReporters[i].ReportLayerFeedSetting(m_nIterateStep, m_nLayer, batch_size, rate);
	}
}

void CKDReporterPool::ReportLayerFeedWeight(CKDTensor acc_delta, CKDTensor weight) {
	if (!m_flagCheck(REPORT_FLAG_LAYER_FEED_WEIGHT)) return;

	for (int i = 0; i < m_nReporterCount; i++) {
		m_pReporters[i].ReportLayerFeedWeight(m_nIterateStep, m_nLayer, acc_delta, weight);
	}
}

void CKDReporterPool::ReportLayerFeedBias(CKDTensor acc_delta, CKDTensor bias) {
	if (!m_flagCheck(REPORT_FLAG_LAYER_FEED_BIAS)) return;

	for (int i = 0; i < m_nReporterCount; i++) {
		m_pReporters[i].ReportLayerFeedBias(m_nIterateStep, m_nLayer, acc_delta, bias);
	}
}

void CKDReporterPool::ReportLayerFeedBias(double acc_delta, CKDTensor bias) {
	if (!m_flagCheck(REPORT_FLAG_LAYER_FEED_BIAS)) return;

	for (int i = 0; i < m_nReporterCount; i++) {
		m_pReporters[i].ReportLayerFeedBias(m_nIterateStep, m_nLayer, acc_delta, bias);
	}
}


void CKDReporterPool::ReportFinalResult(CKDTensor weight, CKDTensor bias) {
	if (!m_flagCheck(REPORT_FLAG_FINAL_RESULT)) return;

	for (int i = 0; i < m_nReporterCount; i++) {
		m_pReporters[i].ReportFinalResult(m_nLayer, weight, bias);
	}
}

void CKDReporterPool::ReportFinalResult(CKDTensor weight1, CKDTensor weight2, CKDTensor bias) {
	if (!m_flagCheck(REPORT_FLAG_FINAL_RESULT)) return;

	for (int i = 0; i < m_nReporterCount; i++) {
		m_pReporters[i].ReportFinalResult(m_nLayer, weight1, weight2, bias);
	}
}

void CKDReporterPool::ReportTestStart(bool test) {
	if (!m_flagCheck(REPORT_FLAG_TEST_START)) return;

	for (int i = 0; i < m_nReporterCount; i++) {
		m_pReporters[i].ReportTestStart(test);
	}
}

void CKDReporterPool::ReportTestCase(CKDReportInfo& info) {
	if (!m_flagCheck(REPORT_FLAG_TEST_CASE)) return;

	for (int i = 0; i < m_nReporterCount; i++) {
		m_pReporters[i].ReportTestCase(info);
	}
}

void CKDReporterPool::ReportTestResult(CKDReportInfo& info) {
	if (!m_flagCheck(REPORT_FLAG_TEST_RESULT)) return;

	for (int i = 0; i < m_nReporterCount; i++) {
		m_pReporters[i].ReportTestResult(info);
	}
}

/*
void CKDReporterPool::ReportTestCase(bool test, int nth, CKDTensor input, CKDTensor output, CKDTensor output_est, bool is_correct) {
	if (!m_flagCheck(REPORT_FLAG_TEST_CASE)) return;

	for (int i = 0; i < m_nReporterCount; i++) {
		m_pReporters[i].ReportTestCase(test, nth, input, output, output_est, is_correct);
	}
}

void CKDReporterPool::ReportTestResult(bool test, int nTestCount, int nCorrectCount) {
	if (!m_flagCheck(REPORT_FLAG_TEST_RESULT)) return;

	for (int i = 0; i < m_nReporterCount; i++) {
		m_pReporters[i].ReportTestResult(test, nTestCount, nCorrectCount);
	}
}
*/