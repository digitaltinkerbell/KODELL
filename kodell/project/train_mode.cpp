#include "stdafx.h"

#include "../project/train_mode.h"
#include "../project/reporter_pool.h"
#include "../project/test_config.h"
#include "../dataset/minibatch.h"
#include "../engine/regular.h"
#include "../util/xml_util.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

CKDTrainMode::CKDTrainMode(XMLNode* node, CKDProjectManager* pManager) {
	m_pManager = pManager;

	m_method = train_sgd;
	m_minitachType = minibatch_random;
	m_iterate = 1;
	m_minibatchSize = 1;

	m_rate = 0;

	m_pReporterPool = NULL;
	m_pTestConfig = NULL;
	m_pRegularPool = NULL;

	XMLNode* train_node = CKDXmlUtil::SeekTagChild(node, "train");

	if (train_node) {
		static const char* method_names[] = { "sg", "sgd", "sgdnm", "rmsprop", "rmsprop_nm", "adagrad", "adadelta", "adam", "newton", NULL };
		m_method = (enumTrainMethod)CKDXmlUtil::GetEnumAttr(train_node, "method", method_names);
		m_iterate = CKDXmlUtil::GetIntAttr(train_node, "iterate");
		m_rate = CKDXmlUtil::GetDoubleAttr(train_node, "rate");

		XMLNode* minibatch_node = CKDXmlUtil::SeekTagChild(train_node, "minibatch");

		if (minibatch_node) {
			static const char* minitach_types[] = { "random", "sequential", "all", "unique", NULL };
			m_minitachType = (enumMinibatchType)CKDXmlUtil::GetEnumAttr(minibatch_node, "type", minitach_types);
			m_minibatchSize = CKDXmlUtil::GetIntAttr(minibatch_node, "size", 1);
		}

		int count = CKDXmlUtil::GetTagChildCount(train_node, "regular");

		if (count > 0) {
			m_pRegularPool = new CKDRegularPool;
			m_pRegularPool->Setup(train_node);

		}

		count = CKDXmlUtil::GetTagChildCount(train_node, "report");

		if (count > 0) {
			m_pReporterPool = new CKDReporterPool;
			m_pReporterPool->Setup(train_node);

		}

		XMLNode* test_node = CKDXmlUtil::SeekTagChild(train_node, "test");
		
		if (test_node) {
			m_pTestConfig = new CKDTestConfig(test_node, pManager);
		}
	}
}

CKDTrainMode::~CKDTrainMode() {
	delete m_pRegularPool;
	delete m_pReporterPool;
	delete m_pTestConfig;
}

int CKDTrainMode::GetTestCount(bool testset) const {
	return m_pTestConfig ? m_pTestConfig->GetTestCount(testset) : -1;
}

/*
int CKDTrainMode::GetTestExampleCount() const {
	return m_pTestConfig ? m_pTestConfig->GetTestExampleCount() : 0;
}

*/

/*
bool CKDTrainMode::GetNthTestExample(int nth, CKDTensor& input, CKDTensor& output) const {
	if (!m_pTestConfig) return false;
	m_pTestConfig->GetNthTestExample(nth, input, output);
	return true;
}
*/

void CKDTrainMode::EvalRegularCost(const CKDTensor &estimate, const CKDTensor &answer, CKDLearningEngine* pEngine, double* pCost) const {
	if (m_pRegularPool) m_pRegularPool->EvalRegularCost(estimate, answer, pEngine, pCost);
}

void CKDTrainMode::EvalRegularGrad(const CKDTensor &estimate, const CKDTensor &answer, CKDLearningEngine* pEngine, CKDTensor* pGrad) const {
	if (m_pRegularPool) m_pRegularPool->EvalRegularGrad(estimate, answer, pEngine, pGrad);
}

double CKDTrainMode::GetWeightDecayNorm2Ratio() const {
	return m_pRegularPool ? m_pRegularPool->GetWeightDecayNorm2Ratio() : 0;
}

CKDTestValidateInfo* CKDTrainMode::GetValidateSettings() {
	return m_pTestConfig ? m_pTestConfig->GetValidateSettings() : NULL;
}
