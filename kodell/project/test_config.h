#pragma once

#include "../util/xml_util.h"

class CKDVector;
class CKDTensor;
class CKDDataSet;
class CKDProjectManager;

struct CKDTestValidateInfo {
	int m_period;
	int m_nTrainCount;
	int m_nTestCount;

	int GetTestCount(bool testset) const { return testset ? m_nTestCount : m_nTrainCount; }
};

class CKDTestConfig {
public:
	CKDTestConfig(XMLNode* node, CKDProjectManager* pManager);
	virtual ~CKDTestConfig();

	//int GetTestExampleCount() const;

	//int GetTestCountForTrain() const { return m_nTrainCount; }
	int GetTestCount(bool testset) const { return testset ? m_nTestCount : m_nTrainCount; }

	//void GetNthTestExample(int nth, CKDTensor& input, CKDTensor& output) const;

	CKDTestValidateInfo* GetValidateSettings();

protected:
	CKDProjectManager* m_pManager;
	CKDTestValidateInfo* m_pValidate;

	int m_nTrainCount;
	int m_nTestCount;
};