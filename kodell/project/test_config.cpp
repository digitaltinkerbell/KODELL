#include "stdafx.h"

#include "../project/test_config.h"
#include "../project/project_manager.h"
#include "../project/reporter_pool.h"
#include "../dataset/dataset.h"
#include "../util/xml_util.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

CKDTestConfig::CKDTestConfig(XMLNode* node, CKDProjectManager* pManager) {
	m_pManager = pManager;
	m_pValidate = NULL;

	m_nTrainCount = CKDXmlUtil::GetIntAttr(node, "trainset", 0);
	m_nTestCount = CKDXmlUtil::GetIntAttr(node, "testset", 0);

	XMLNode* validate_node = CKDXmlUtil::SeekTagChild(node, "validate");

	if (validate_node) {
		m_pValidate = new CKDTestValidateInfo;
		m_pValidate->m_period = CKDXmlUtil::GetIntAttr(validate_node, "period");
		m_pValidate->m_nTrainCount = CKDXmlUtil::GetIntAttr(validate_node, "trainset", 0);
		m_pValidate->m_nTestCount = CKDXmlUtil::GetIntAttr(validate_node, "testset", 0);
	}
}

CKDTestConfig::~CKDTestConfig() {
	delete m_pValidate;
}

/*
int CKDTestConfig::GetTestExampleCount() const {
	if (m_nTestCount > 0) return m_nTestCount;
	return m_pManager->GetTestExampleCount();
}
*/

/*
void CKDTestConfig::GetNthTestExample(int nth, CKDTensor& input, CKDTensor& output) const {
	m_pManager->GetNthTestExample(nth, input, output);
}
*/

CKDTestValidateInfo* CKDTestConfig::GetValidateSettings() {
	return m_pValidate;
}
