#include "stdafx.h"

#include <string.h>
#include <math.h>

#include "../engine/learning_engine.h"
#include "../engine/layer.h"
#include "../engine/context_stack.h"
#include "../project/project_manager.h"
#include "../project/network_model.h"
#include "../project/train_mode.h"
#include "../project/reporter_pool.h"
#include "../project/test_config.h"
#include "../dataset/minibatch.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

CKDLearningEngine::CKDLearningEngine(CKDProjectManager* pConfig) {
	m_pConfig = pConfig;

	m_nLayerCount = 0;
	m_ppLayers = NULL;
}

CKDLearningEngine::~CKDLearningEngine() {
	if (m_ppLayers) {
		for (int i = 0; i < m_nLayerCount; i++) {
			delete m_ppLayers[i];
		}
		delete [] m_ppLayers;
	}
}

void CKDLearningEngine::Setup() {
	CKDNetworkModel* pModel = m_pConfig->GetNetworkModel();

	if (m_nLayerCount || m_ppLayers) THROW_TEMP;

	int nActualLayerCount = pModel->GetActualLayerCount();
	int nLayerInfoCount = pModel->GetLayerInfoCount();

	m_ppLayers = new CKDLayer* [nActualLayerCount];
	memset(m_ppLayers, 0, sizeof(CKDLayer*)*nActualLayerCount);

	CKDDimension inputDim = m_pConfig->GetInputDimension();
	CKDDimension outputDim = m_pConfig->GetOutputDimension();

	if (inputDim.isEmpty()) THROW_TEMP;

	while (--nLayerInfoCount >= 0) {
		const CKDLayerConfig* pLayerConfig = pModel->GetNthLayerInfo(nLayerInfoCount);

		int depth = pLayerConfig->GetDepth();

		for (int i = 0; i < depth; i++) {
			CKDLayer* pLayer = CKDLayer::Create(pLayerConfig, inputDim, outputDim);
			inputDim = pLayer->GetOutputDimension();
			m_ppLayers[m_nLayerCount++] = pLayer;
		}
	}
}

void CKDLearningEngine::Train() {
	enumTrainMethod train_method = m_pConfig->GetTrainMethod();
	
	if (train_method == train_undef) train_method = train_sgd;
	if (train_method != train_sgd) THROW_TEMP;

	int nIterate = m_pConfig->GetIterateCount();
	int nContStaurate = 0;

	CKDMinibatchContext context;
	CKDReporterPool* pReporters = m_pConfig->GetReporterPool();

	pReporters->ReportTrainStart(m_pConfig->GetProjectName());

	CKDTestValidateInfo* pValidate = m_pConfig->GetValidateSettings();

	int i = 0;
	for (i = 0; i < nIterate; i++) {
		pReporters->ReportIterateStepStart(i);

		if (i > 0 && pValidate && i % pValidate->m_period == 0) {
			m_validate(i, pValidate, pReporters);
		}

		CKDMinibatch minibatch(m_pConfig, &context, pReporters);
		m_resetBackGradient();

		int size = minibatch.size();

		enum enumProjectType type = m_pConfig->GetProjectType();
		enum enumProjectMode mode = m_pConfig->GetProjectMode();

		CKDContextStack context(this, type, mode, test_dest_train, size);

		for (int k = 0; k < size; k++) {
			CKDTensor input, output;
			minibatch.GetExample(k, &input, &output);
			context.setExample(input, output, k);

			while (context.get_reccurent_next_step()) {
				m_execForward_step(context, pReporters);
			}

			while (context.get_reccurent_prev_step()) {
				m_execBackward_step(context, pReporters);
			}
		}

		bool all_near_zero = m_feedGradient(context, m_pConfig->GetLearningRate(), minibatch.size(), pReporters);
		if (all_near_zero) {
			if (++nContStaurate >= 10) {
				//pReporters->ReportTrainExit("exit on saturate on continuous 3 times");
				//printf("exit on saturate: iterate count = %d\n", i + 1);
				//break;
			}
		}
		else {
			nContStaurate = 0;
		}
		pReporters->ReportIterateStepEnd();
	}

	m_finalReport(pReporters);
	pReporters->ReportTrainEnd(i);
}

double CKDLearningEngine::EvalRegularCost(const CKDTensor &estimate, const CKDTensor &answer) {
	double cost = 0;
	m_pConfig->EvalRegularCost(estimate, answer, this, &cost);
	return cost;
}

CKDTensor CKDLearningEngine::EvalRegularGrad(const CKDTensor &estimate, const CKDTensor &answer) {
	CKDTensor grad(estimate.get_dimension());
	m_pConfig->EvalRegularGrad(estimate, answer, this, &grad);
	return grad;
}

void CKDLearningEngine::m_execForward_step(CKDContextStack& context, CKDReporterPool* pReporters) {
	CKDTensor input = context.getStepInput();
	CKDTensor output = context.getStepOutput();

	pReporters->ReportExample(input, output);

	for (int i = 0; i < m_nLayerCount; i++) {
		context.pushLayer(m_ppLayers[i]);

		pReporters->ReportSetLayer(i, m_nLayerCount, train_phase_forward);
		m_ppLayers[i]->ExecFowardStep(context, pReporters);
	}

	double cost = context.eval_cost();
	pReporters->ReportCost(cost, context.need_report_reset(), context.need_report_close());
	CKDTensor cost_grad = context.eval_cost_grad();
	context.setCostGradient(cost_grad);
	pReporters->ReportCostGradient(cost_grad);
}

void CKDLearningEngine::m_execBackward_step(CKDContextStack& context, CKDReporterPool* pReporters) {
	for (int i = m_nLayerCount - 1; i >= 0; i--) {
		context.setLayer(m_ppLayers[i]);
		pReporters->ReportSetLayer(i, m_nLayerCount, train_phase_backward);
		m_ppLayers[i]->ExecBackwardStep(context, pReporters);
	}
}

void CKDLearningEngine::m_resetBackGradient() {
	for (int i = m_nLayerCount - 1; i >= 0; i--) {
		m_ppLayers[i]->ResetBackGradient();
	}
}

bool CKDLearningEngine::m_feedGradient(CKDContextStack& context, double rate, int batch_size, CKDReporterPool* pReporters) {
	bool all_near_zero = true;
	for (int i = m_nLayerCount - 1; i >= 0; i--) {
		context.setLayer(m_ppLayers[i]);
		pReporters->ReportSetLayer(i, m_nLayerCount, train_phase_feed);
		bool near_zero = m_ppLayers[i]->FeedGradient(context, rate, batch_size, pReporters, m_pConfig);
		if (!near_zero) all_near_zero = false;
	}

	return all_near_zero;
}

void CKDLearningEngine::m_finalReport(CKDReporterPool* pReporters) {
	for (int i = 0; i < m_nLayerCount; i++) {
		pReporters->ReportSetLayer(i, m_nLayerCount, train_phase_final);
		m_ppLayers[i]->SaveFinalResult(pReporters);
	}
}

void CKDLearningEngine::Test() {
	m_gradeTestExamples(NULL, false);
	m_gradeTestExamples(NULL, true);
}

void CKDLearningEngine::m_validate(int step, CKDTestValidateInfo* pValidate, CKDReporterPool* pReporters) {
	m_gradeTestExamples(pValidate, false);
	m_gradeTestExamples(pValidate, true);
}

void CKDLearningEngine::m_gradeTestExamples(CKDTestValidateInfo* pValidate, bool testset) {
	CKDReporterPool* pReporters = m_pConfig->GetReporterPool();

	pReporters->ReportTestStart(testset);

	int total_count = m_pConfig->GetTestExampleCount(testset);
	int test_count = pValidate ? pValidate->GetTestCount(testset) : m_pConfig->GetTestCount(testset);
	int correct_count = 0;

	if (test_count < 0) return;

	if (test_count == 0) {
		if (pValidate) return;
		test_count = total_count;
	}

	enum enumProjectType type = m_pConfig->GetProjectType();
	enum enumProjectMode mode = m_pConfig->GetProjectMode();

	CKDContextStack context(this, type, mode, testset ? test_dest_testset : test_dest_trainset, test_count);

	for (int n = 0; n < test_count; n++) {
		CKDTensor input, output;

		int nth = (total_count == test_count) ? n : rand() % total_count;

		if (!m_pConfig->GetNthTestExample(testset, nth, input, output)) THROW_TEMP;

		context.setExample(input, output, n);

		while (context.get_reccurent_next_step()) {
			m_execForward_step(context, pReporters);

			CKDReportInfo reportInfo;
			context.gradeTestResult(reportInfo);
			pReporters->ReportTestCase(reportInfo);
		}
	}

	CKDReportInfo reportFullInfo;
	context.getFullGradeResult(reportFullInfo);
	pReporters->ReportTestResult(reportFullInfo);
}

double CKDLearningEngine::GetWeightSqSum() {
	double sqsum = 0;

	for (int i = 0; i < m_nLayerCount; i++) {
		sqsum += m_ppLayers[i]->GetWeightSqSum();
	}

	return sqsum;
}

double CKDLearningEngine::GetWeightAbsSum() {
	double sum = 0;

	for (int i = 0; i < m_nLayerCount; i++) {
		sum += m_ppLayers[i]->GetWeightAbsSum();
	}

	return sum;
}