#include "stdafx.h"

#include <math.h>

#include "../engine/context_stack.h"
#include "../engine/layer.h"
#include "../engine/learning_engine.h"
#include "../project/project_manager.h"
#include "../project/reporter.h"
#include "../math/tensor.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

CKDContextStack::CKDContextStack(CKDLearningEngine* pEngine, enum enumProjectType type, enum enumProjectMode mode, enum enumTestDest dest, int iterate) {
	m_pEngine = pEngine;

	m_projectType = type;
	m_projectMode = mode;
	m_testDest = dest;
	m_iterate = iterate;

	m_layerListHead = NULL;
	m_layerListTail = NULL;
	m_currLayer = NULL;

	m_testCount = 0;
	m_correctCount = 0;
}

CKDContextStack::~CKDContextStack() {
	m_resetLayerList();
}

void CKDContextStack::setExample(CKDTensor input, CKDTensor output, int nth) {
	m_exampleInput = input;
	m_exampleOutput = output;
	m_exampleIndex = nth;

	m_sequenceLength = (m_projectType != proj_type_recurrent) ? 1 : input.size(0);
	m_sequenceIndex = -1;
}

void CKDContextStack::m_resetLayerList() {
	delete m_layerListHead;

	m_layerListHead = NULL;
	m_layerListTail = NULL;
	m_currLayer = NULL;
}

bool CKDContextStack::get_reccurent_next_step() {
	return ++m_sequenceIndex < m_sequenceLength;
}

bool CKDContextStack::get_reccurent_prev_step() {
	return --m_sequenceIndex >= 0;
}

bool CKDContextStack::need_report_reset() {
	return m_sequenceIndex == 0;
}

bool CKDContextStack::need_report_close() {
	//return m_sequenceIndex == m_sequenceLength - 1;
	return m_exampleIndex == m_iterate - 1;
}


double CKDContextStack::eval_cost() {
	CKDTensor estimate = m_layerListTail->getTensor("__result__", m_sequenceIndex);
	CKDTensor answer = (m_projectType != proj_type_recurrent) ? m_exampleOutput : m_exampleOutput.nthComponent(m_sequenceIndex);

	return m_eval_pure_cost(estimate, answer) + m_eval_regular_cost(estimate, answer);
}

double CKDContextStack::m_eval_pure_cost(const CKDTensor &estimate, const CKDTensor &answer) {
	enum costFuncType funcType = m_layerListTail->m_pLayer->GetCostFunc();

	if (funcType == cost_func_mse) {
		CKDTensor diff = estimate - answer;
		CKDTensor square = diff.square();

		return square.sum() / estimate.size();
	}
	else if (funcType == cost_func_cross_entropy) {
		if (m_projectMode == proj_mode_regression) {
			CKDTensor diff = estimate - answer;
			CKDTensor square = diff.square();

			return square.sum() / estimate.size();
		}
		else if (m_projectMode == proj_mode_binary) {
			if (estimate.size() != 1) THROW_TEMP;
			if (answer.size() != 1) THROW_TEMP;

			if (answer[0] < 0.5) return log(1 + exp(estimate[0]));
			else return log(1 + exp(-estimate[0]));
		}
		else if (m_projectMode == proj_mode_classify) {
			int answer_idx = answer.get_max_index();
			double prob = estimate[answer_idx];
			return -log(estimate[answer_idx]);
		}
		else {
			THROW_TEMP;
		}
	}
	else {
		THROW_TEMP;
	}

	return 0;
}

double CKDContextStack::m_eval_regular_cost(const CKDTensor &estimate, const CKDTensor &answer) {
	return m_pEngine->EvalRegularCost(estimate, answer);
}

CKDTensor CKDContextStack::getStepInput() {
	if (m_projectType != proj_type_recurrent) {
		return m_exampleInput;
	}
	else {
		return m_exampleInput.nthComponent(m_sequenceIndex);
	}
}

CKDTensor CKDContextStack::getStepOutput() {
	if (m_projectType != proj_type_recurrent) {
		return m_exampleOutput;
	}
	else {
		return m_exampleOutput.nthComponent(m_sequenceIndex);
	}
}

void CKDContextStack::pushLayer(CKDLayer* pLayer) {
	CKDLayerInfo* pNode = m_layerListHead->seekLayerNode(pLayer);

	if (pNode == NULL) {
		pNode = new CKDLayerInfo(pLayer);

		pNode->m_pPrevNode = m_layerListTail;

		if (m_layerListHead) m_layerListTail->m_pNextNode = pNode;
		else m_layerListHead = pNode;

		m_layerListTail = pNode;
	}

	CKDTensor tensor;

	if (pNode->m_pPrevNode) tensor = pNode->m_pPrevNode->getTensor("__result__", m_sequenceIndex);
	else if ((m_projectType != proj_type_recurrent)) tensor = m_exampleInput;
	else tensor = m_exampleInput.nthComponent(m_sequenceIndex);

	pNode->setTensor("__input__", tensor, m_sequenceIndex);
	m_currLayer = pNode;
}

void CKDContextStack::setLayer(CKDLayer* pLayer) {
	CKDLayerInfo* pNode = m_layerListHead->seekLayerNode(pLayer);
	if (pNode == NULL) THROW_TEMP;
	m_currLayer = pNode;
}


CKDTensor CKDContextStack::eval_cost_grad() {
	CKDTensor estimate = m_layerListTail->getTensor("__result__", m_sequenceIndex);
	CKDTensor answer = (m_projectType != proj_type_recurrent) ? m_exampleOutput : m_exampleOutput.nthComponent(m_sequenceIndex);

	m_currGradient = m_eval_pure_cost_grad(estimate, answer) + m_eval_regular_cost_grad(estimate, answer);

	return m_currGradient;
}

CKDTensor CKDContextStack::m_eval_pure_cost_grad(const CKDTensor &estimate, const CKDTensor &answer) {
	CKDDimension dimension = answer.get_dimension();

	if (estimate.get_dimension() != dimension) THROW_TEMP;

	CKDTensor result(dimension);

	enum costFuncType funcType = m_layerListTail->m_pLayer->GetCostFunc();

	if (funcType == cost_func_mse) {
		result = estimate - answer;
	}
	else if (funcType == cost_func_cross_entropy) {
		if (m_projectMode == proj_mode_regression) {
			result = estimate - answer;
		}
		else if (m_projectMode == proj_mode_binary) {
			if (result.size() != 1) THROW_TEMP;

			if (answer[0] < 0.5) result[0] = 1 / (1 + exp(-estimate[0]));
			else result[0] = -1 / (1 + exp(estimate[0]));
		}
		else if (m_projectMode == proj_mode_classify) {
			for (int i = 0; i < result.size(); i++) {
				result[i] = (answer[i] > 0.5) ? (-1 / estimate[i]) : 0;
			}
			return result;
		}
		else {
			THROW_TEMP;
		}
	}
	else {
		THROW_TEMP;
	}

	return result;
}

CKDTensor CKDContextStack::m_eval_regular_cost_grad(const CKDTensor &estimate, const CKDTensor &answer) {
	return m_pEngine->EvalRegularGrad(estimate, answer);
}

void CKDContextStack::gradeTestResult(CKDReportInfo& info) {
//bool CKDProjectManager::GradeTestResult(CKDTensor estimate, CKDTensor output) {
	CKDTensor estimate = m_layerListTail->getTensor("__result__", m_sequenceIndex);
	CKDTensor answer = (m_projectType != proj_type_recurrent) ? m_exampleOutput : m_exampleOutput.nthComponent(m_sequenceIndex);

	bool is_correct = false;

	if (m_projectMode == proj_mode_regression) {
		is_correct =(estimate - answer).all_near_zero();
	}
	else if (m_projectMode == proj_mode_binary) {
		if (estimate.size() != 1) THROW_TEMP;
		if (answer.size() != 1) THROW_TEMP;

		if (answer[0] > 0.5) is_correct = estimate[0] > 0.99;
		else is_correct = estimate[0] < 0.01;
	}
	else if (m_projectMode == proj_mode_classify) {
		if (estimate.size() != answer.size()) THROW_TEMP;

		int nEstIndex = estimate.get_max_index();
		int nAnsIndex = answer.get_max_index();

		is_correct = (nEstIndex == nAnsIndex);
	}
	else {
		THROW_TEMP;
	}

	m_testCount++;
	if (is_correct) {
		m_correctCount++;
	}

	info.setValue("is_test", m_testDest == test_dest_testset);
	info.setValue("nth", m_testCount);
	info.setValue("is_correct", is_correct);
}

void CKDContextStack::getFullGradeResult(CKDReportInfo& info) {
	info.setValue("is_test", m_testDest == test_dest_testset);
	info.setValue("test_count", m_testCount);
	info.setValue("correct_count", m_correctCount);
}


bool CKDContextStack::get_need_back() {
	THROW_TEMP;
}

bool CKDContextStack::get_need_input_grad() {
	return m_currLayer != m_layerListHead;
}

bool CKDContextStack::get_need_recur_grad() {
	return m_sequenceIndex > 0;
}

CKDTensor CKDContextStack::getInput(int nth) {
	return m_currLayer->getTensor("__input__", m_sequenceIndex+nth);
}

CKDTensor CKDContextStack::getGradient(int nth) {
	return m_currLayer->getTensor("__gradient__", m_sequenceIndex + nth);
}

CKDTensor CKDContextStack::getLayerTensorXXX(const char* pKey) {
	return m_currLayer->getTensor(pKey, -1);
}

double CKDContextStack::getLayerScalarXXX(const char* pKey) {
	CKDTensor tensor = m_currLayer->getTensor(pKey, -1);

	if (tensor.size() != 1) THROW_TEMP;

	return tensor[0];
}

CKDTensor CKDContextStack::getInstantTensor(const char* pKey, int nth) {
	if (m_sequenceIndex + nth >= m_sequenceLength) return CKDTensor();
	return m_currLayer->getTensor(pKey, m_sequenceIndex + nth);
}

CKDTensor CKDContextStack::getOutput(int nth) {
	if (m_sequenceIndex + nth < 0) return CKDTensor();
	return m_currLayer->getTensor("__result__", m_sequenceIndex + nth);
}

void CKDContextStack::setOutput(CKDTensor result, int nth) {
	m_currLayer->setTensor("__result__", result, m_sequenceIndex + nth);
}

void CKDContextStack::setGradient(CKDTensor result, int nth) {
	m_currLayer->m_pPrevNode->setTensor("__gradient__", result, m_sequenceIndex + nth);
}

void CKDContextStack::setCostGradient(CKDTensor result, int nth) {
	m_currLayer->setTensor("__gradient__", result, m_sequenceIndex + nth);
}

void CKDContextStack::setLayerTensorXXX(const char* pKey, CKDTensor tensor) {
	m_currLayer->setTensor(pKey, tensor, -1);
}

void CKDContextStack::setInstantTensor(const char* pKey, CKDTensor tensor, int nth) {
	m_currLayer->setTensor(pKey, tensor, m_sequenceIndex + nth);
}

void CKDContextStack::accumulateLayerTensorXXX(const char* pKey, CKDTensor tensor) {
	CKDTensorMapNode* pNode = m_currLayer->m_pMapHead->seekNode(pKey, -1);
	if (pNode) {
		pNode->m_tensor += tensor;
	}
	else {
		m_currLayer->setTensor(pKey, tensor, -1);
	}
}

CKDLayerInfo::CKDLayerInfo(CKDLayer* pLayer) {
	m_pLayer = pLayer;

	m_pPrevNode = NULL;
	m_pNextNode = NULL;

	m_pMapHead = NULL;
	m_pMapTail = NULL;
}

CKDLayerInfo::~CKDLayerInfo() {
	delete m_pNextNode;
	delete m_pMapHead;
}

CKDTensor CKDLayerInfo::getTensor(const char* pName, int index) {
	CKDTensorMapNode* pNode = m_pMapHead->seekNode(pName, index);
	if (!pNode) THROW_TEMP;
	return pNode->m_tensor;
}

/*
CKDTensor CKDLayerInfo::getRecurrentTensorOnTop(const char* pName) {
	CKDTensorMapNode* pNode = m_pMapTail->seekReverseNode(pName);
	if (!pNode) return CKDTensor();
	return pNode->m_tensor;
}
*/

void CKDLayerInfo::setTensor(const char* pName, CKDTensor tensor, int index) {
	CKDTensorMapNode* pNode = m_pMapHead->seekNode(pName, index);
	if (pNode) {
		pNode->m_tensor = tensor;
	}
	else {
		CKDTensorMapNode* p = new CKDTensorMapNode(pName, index, tensor);

		p->m_pPrevNode = m_pMapTail;

		if (m_pMapHead) m_pMapTail->m_pNextNode = p;
		else m_pMapHead = p;

		m_pMapTail = p;
	}
}

/*
void CKDLayerInfo::pushTensor(const char* pName, CKDTensor tensor) {
	CKDTensorMapNode* p = new CKDTensorMapNode(pName, tensor);

	p->m_pPrevNode = m_pMapTail;

	if (m_pMapHead) m_pMapTail->m_pNextNode = p;
	else m_pMapHead = p;

	m_pMapTail = p;
}
*/

CKDLayerInfo* CKDLayerInfo::seekLayerNode(CKDLayer* pLayer) {
	if (this == NULL) return NULL;
	if (m_pLayer == pLayer) return this;
	return m_pNextNode->seekLayerNode(pLayer);
}

CKDTensorMapNode::CKDTensorMapNode(const char* pName, int index, CKDTensor tensor) {
	m_pName = pName;
	m_tensor = tensor;
	m_index = index;

	m_pPrevNode = NULL;
	m_pNextNode = NULL;
}

CKDTensorMapNode::~CKDTensorMapNode() {
	delete m_pNextNode;
}

CKDTensorMapNode* CKDTensorMapNode::seekNode(const char* pName, int index) {
	if (this == NULL) return NULL;
	if (strcmp(m_pName, pName) == 0 && m_index == index) return this;
	return m_pNextNode->seekNode(pName, index);
}

CKDTensorMapNode* CKDTensorMapNode::seekReverseNode(const char* pName, int index) {
	if (this == NULL) return NULL;
	if (strcmp(m_pName, pName) == 0 && m_index == index) return this;
	return m_pPrevNode->seekNode(pName, index);
}

/*
double CKDLearningEngine::m_eval_cost(const CKDTensor &estimate, const CKDTensor &answer) {
	return m_eval_pure_cost(estimate, answer) + m_eval_regular_cost(estimate, answer);
}

double CKDLearningEngine::m_eval_pure_cost(const CKDTensor &estimate, const CKDTensor &answer) {
	enum costFuncType funcType = m_ppLayers[m_nLayerCount - 1]->GetCostFunc();

	if (funcType == cost_func_mse) {
		CKDTensor diff = estimate - answer;
		CKDTensor square = diff.square();

		return square.sum() / estimate.size();
	}
	else if (funcType == cost_func_cross_entropy) {
		enumProjectMode project_mode = m_pConfig->GetProjectMode();

		if (project_mode == proj_mode_regression) {
			CKDTensor diff = estimate - answer;
			CKDTensor square = diff.square();

			return square.sum() / estimate.size();
		}
		else if (project_mode == proj_mode_binary) {
			if (estimate.size() != 1) THROW_TEMP;
			if (answer.size() != 1) THROW_TEMP;

			if (answer[0] < 0.5) return log(1 + exp(estimate[0]));
			else return log(1 + exp(-estimate[0]));
		}
		else if (project_mode == proj_mode_classify) {
			int answer_idx = answer.get_max_index();
			double prob = estimate[answer_idx];
			return -log(estimate[answer_idx]);

			/*
			int size = answer.size();
			if (estimate.size() != size) THROW_TEMP;

			int answer_idx = answer.get_max_index();
			double prob = estimate[answer_idx];
			return -log(estimate[answer_idx]);
			for (int i = 0; i < size; i++) {
			double prob = estimate[answer_idx];
			return -log(estimate[answer_idx]);
			}
			*/
			/*
			int size = answer.size();

			if (estimate.size() != size) THROW_TEMP;

			CKDTensor result(size);

			for (int i = 0; i < size; i++) {
			if (answer[i] < 0.5) {
			result[i] = log(1 - estimate[i]);
			//result[i] = log(1 + exp(estimate[i]));
			}
			else {
			result[i] = log(estimate[i]);
			//result[i] = log(1 + exp(-estimate[i]));
			}
			}

			return -result.sum() / size;
			*/ /*
		}
		else {
			THROW_TEMP;
		}
	}
	else {
		THROW_TEMP;
	}

	return 0;
}

double CKDLearningEngine::m_eval_regular_cost(const CKDTensor &estimate, const CKDTensor &answer) {
	double cost = 0;
	m_pConfig->EvalRegularCost(estimate, answer, this, &cost);
	return cost;
}

CKDTensor CKDLearningEngine::m_eval_cost_grad(const CKDTensor &estimate, const CKDTensor &answer) {
	return m_eval_pure_cost_grad(estimate, answer) + m_eval_regular_cost_grad(estimate, answer);
}

CKDTensor CKDLearningEngine::m_eval_pure_cost_grad(const CKDTensor &estimate, const CKDTensor &answer) {
	CKDDimension dimension = answer.get_dimension();

	if (estimate.get_dimension() != dimension) THROW_TEMP;

	CKDTensor result(dimension);

	enum costFuncType funcType = m_ppLayers[m_nLayerCount - 1]->GetCostFunc();

	if (funcType == cost_func_mse) {
		result = estimate - answer;
	}
	else if (funcType == cost_func_cross_entropy) {
		enumProjectMode project_mode = m_pConfig->GetProjectMode();

		if (project_mode == proj_mode_regression) {
			result = estimate - answer;
		}
		else if (project_mode == proj_mode_binary) {
			if (result.size() != 1) THROW_TEMP;

			if (answer[0] < 0.5) result[0] = 1 / (1 + exp(-estimate[0]));
			else result[0] = -1 / (1 + exp(estimate[0]));
		}
		else if (project_mode == proj_mode_classify) {
			/*
			for (int i = 0; i < size; i++) {
			for (int j = 0; j < size; j++) {
			double log_diff = ((i == j) ? 1 : 0) - estimate[j];

			if (answer[j] < 0.5) {
			result[i] -= log_diff;
			}
			else {
			result[i] += log_diff;
			}
			}
			//result[i] *= log(estimate[i]);
			}
			return result * (-1.0 / size);
			*/ /*
			for (int i = 0; i < result.size(); i++) {
				result[i] = (answer[i] > 0.5) ? (-1 / estimate[i]) : 0;
			}
			return result;
		}
		else {
			THROW_TEMP;
		}
	}
	else {
		THROW_TEMP;
	}

	return result;
}

CKDTensor CKDLearningEngine::m_eval_regular_cost_grad(const CKDTensor &estimate, const CKDTensor &answer) {
	CKDTensor grad(estimate.get_dimension());
	m_pConfig->EvalRegularGrad(estimate, answer, this, &grad);
	return grad;
}


void CKDLearningEngine::m_gradeTestExamples(CKDTestValidateInfo* pValidate) {
	CKDReporterPool* pReporters = m_pConfig->GetReporterPool();

	pReporters->ReportTestStart(true);

	int total_count = m_pConfig->GetTestExampleCount();
	int test_count = pValidate ? pValidate->m_nTestCount : m_pConfig->GetTestCountForTest();
	int correct_count = 0;

	if (test_count == 0) {
		if (pValidate) return;
		test_count = total_count;
	}

	for (int n = 0; n < test_count; n++) {
		CKDTensor input;
		CKDTensor output;
		CKDTensor net;
		CKDTensor layer_input;
		CKDTensor layer_output;
		CKDTensor empty;

		int nth = (total_count == test_count) ? n : rand() % total_count;

		if (!m_pConfig->GetNthTestExample(nth, input, output)) THROW_TEMP;

		layer_input = input;

		for (int i = 0; i < m_nLayerCount; i++) {
			layer_output = m_ppLayers[i]->ExecFowardStep(false, layer_input, empty, net, pReporters);
			layer_input = layer_output;
		}

		bool is_correct = m_pConfig->GradeTestResult(layer_output, output);

		if (is_correct) correct_count++;

		pReporters->ReportTestCase(true, n, input, output, layer_output, is_correct);
	}

	pReporters->ReportTestResult(true, test_count, correct_count);
}

*/
