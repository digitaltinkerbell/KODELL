#include "stdafx.h"

#include "../project/project_manager.h"
#include "../config/config_manager.h"
#include "../project/train_mode.h"
#include "../dataset/dataset.h"
#include "../dataset/minibatch.h"
#include "../engine/learning_engine.h"
#include "../util/xml_util.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

CKDProjectManager::CKDProjectManager(CKDConfigManager* pConfMan, const char* name) {
	m_pBaseProject = NULL;
	m_pDataSet = NULL;
	m_pNetworkModel = NULL;

	m_projectNode = pConfMan->LookupProject(name);

	if (m_projectNode == NULL) THROW_TEMP;

	m_pProjectName = name;
	m_pProjectDesc = NULL;
	m_pDummyMsg = CKDXmlUtil::GetAttribute(m_projectNode, "dummy");

	if (m_pDummyMsg != NULL) return;

	XMLNode* desc_node = CKDXmlUtil::SeekTagChild(m_projectNode, "description");
	
	if (desc_node) {
		const char* desc = desc_node->ToElement()->GetText();
		printf("%s\n", desc);
	}
	
	const char*base_name = m_projectNode->ToElement()->Attribute("base");

	m_pBaseProject = base_name ? (new CKDProjectManager(pConfMan, base_name)) : NULL;

	static const char* mode_names[] = { "regression", "binary", "classify", NULL };
	static const char* mode_types[] = { "normal", "recurrent", NULL };

	m_projectMode = (enum enumProjectMode) CKDXmlUtil::GetEnumAttr(m_projectNode, "mode", mode_names);
	m_projectType = (enum enumProjectType) CKDXmlUtil::GetEnumAttr(m_projectNode, "type", mode_types);

	if (m_projectMode == proj_mode_undef && m_pBaseProject) m_projectMode = m_pBaseProject->GetProjectMode();
	if (m_projectMode == proj_mode_undef) THROW_TEMP;

	if (m_projectType == proj_type_undef  && m_pBaseProject) m_projectType = m_pBaseProject->GetProjectType();
	if (m_projectType == proj_type_undef) m_projectType = proj_type_normal;

	m_pDataSet = CKDDataSet::Create(this, m_projectNode);
	m_pNetworkModel = CKDNetworkModel::Create(m_projectNode);
	m_pTrainMode = new CKDTrainMode(m_projectNode, this);
}

CKDProjectManager::~CKDProjectManager() {
	delete m_pBaseProject;
	delete m_pDataSet;
	delete m_pNetworkModel;
	delete m_pTrainMode;
	delete m_pProjectDesc;
}

void CKDProjectManager::Execute() {
	if (m_pDummyMsg) {
		printf("    DUMMY PROJECT [%s] %s\n", m_pProjectName, m_pDummyMsg);
		return;
	}
	
	CKDLearningEngine engine(this);

	engine.Setup();
	engine.Train();
	engine.Test();
}

double CKDProjectManager::GetLearningRate() const {
	double rate = m_pTrainMode ? m_pTrainMode->GetLearningRate() : 0;
	if (!rate && m_pBaseProject) rate = m_pBaseProject->GetLearningRate();
	if (!rate) rate = 0.001;

	return rate;
}

CKDDimension CKDProjectManager::GetInputDimension() const {
	CKDDimension dim;
	if (m_pDataSet) dim = m_pDataSet->GetInputDimension();
	if (dim.isEmpty() && m_pBaseProject) dim = m_pBaseProject->GetInputDimension();

	return dim;
}

CKDDimension CKDProjectManager::GetOutputDimension() const {
	CKDDimension dim;
	if (m_pDataSet) dim = m_pDataSet->GetOutputDimension();
	if (dim.isEmpty() && m_pBaseProject) dim = m_pBaseProject->GetOutputDimension();

	return dim;
}

/*
int CKDProjectManager::GetHiddenDepth() const {
	int depth = m_pNetworkModel ? m_pNetworkModel->GetHiddenDepth() : 0;
	if (!depth && m_pBaseProject) depth = m_pBaseProject->GetHiddenDepth();

	return depth;
}

const char* CKDProjectManager::GetOutputBias() const {
	const char* bias = m_pNetworkModel ? m_pNetworkModel->GetOutputBias() : NULL;
	if (!bias && m_pBaseProject) return m_pBaseProject->GetOutputBias();

	return bias;
}

const char* CKDProjectManager::GetOutputActivateFunc() const {
	const char* func = m_pNetworkModel ? m_pNetworkModel->GetOutputActivateFunc() : NULL;
	if (!func && m_pBaseProject) return m_pBaseProject->GetOutputActivateFunc();

	return func;
}

const char* CKDProjectManager::GetOutputCostFunc() const {
	const char* func = m_pNetworkModel ? m_pNetworkModel->GetOutputCostFunc() : NULL;
	if (!func && m_pBaseProject) return m_pBaseProject->GetOutputCostFunc();

	return func;
}

int CKDProjectManager::GetLayerWidth(int nLayer) const {
	if (nLayer == GetHiddenDepth()) return GetOutputSize();

	int width = m_pNetworkModel ? m_pNetworkModel->GetHiddenWidth(nLayer) : 0;
	if (!width && m_pBaseProject) return m_pBaseProject->GetLayerWidth(nLayer);

	return width;
}

const char* CKDProjectManager::GetLayerActivateFunc(int nLayer) const {
	if (nLayer == GetHiddenDepth()) return GetOutputActivateFunc();

	const char* func = m_pNetworkModel ? m_pNetworkModel->GetHiddenActivateFunc(nLayer) : NULL;
	if (!func && m_pBaseProject) return m_pBaseProject->GetLayerActivateFunc(nLayer);

	return func;
}

const char* CKDProjectManager::GetLayerBias(int nLayer) const {
	if (nLayer == GetHiddenDepth()) return GetOutputBias();

	const char* bias = m_pNetworkModel ? m_pNetworkModel->GetHiddenBias(nLayer) : NULL;
	if (!bias && m_pBaseProject) return m_pBaseProject->GetLayerBias(nLayer);

	return bias;
}

const CKDParameterInitValues* CKDProjectManager::GetLayerInitValue(int nLayer) const {
	const CKDParameterInitValues* pInfo = m_pNetworkModel ? m_pNetworkModel->GetLayerInitValue(nLayer) : NULL;
	if (!pInfo && m_pBaseProject) return m_pBaseProject->GetLayerInitValue(nLayer);

	return pInfo;
}
*/

enumTrainMethod CKDProjectManager::GetTrainMethod() const {
	enumTrainMethod method = m_pTrainMode ? m_pTrainMode->GetTrainMethod() : train_undef;
	if (method == train_undef && m_pBaseProject) return m_pBaseProject->GetTrainMethod();

	return method;
}

int CKDProjectManager::GetIterateCount() const {
	int count = m_pTrainMode ? m_pTrainMode->GetIterateCount() : 0;
	if (!count && m_pBaseProject) return m_pBaseProject->GetIterateCount();

	return count;
}

/*
int CKDProjectManager::GetReportFlags(int nStep) const {
	int flags = m_pTrainMode ? m_pTrainMode->GetReportFlags(nStep) : 0;
	if (!flags && m_pBaseProject) return m_pBaseProject->GetReportFlags(nStep);

	return flags;
}
*/

CKDDataSet* CKDProjectManager::GetDataSet() const {
	CKDDataSet* pDataSet = m_pDataSet;
	if (!pDataSet && m_pBaseProject) return m_pBaseProject->GetDataSet();

	return pDataSet;
}

CKDReporterPool* CKDProjectManager::GetReporterPool() const {
	CKDReporterPool* pReporterPool = m_pTrainMode ? m_pTrainMode->GetReporterPool() : NULL;
	if (!pReporterPool && m_pBaseProject) return m_pBaseProject->GetReporterPool();

	return pReporterPool;
}

CKDNetworkModel* CKDProjectManager::GetNetworkModel() const {
	CKDNetworkModel* pModel = m_pNetworkModel;
	if (!pModel && m_pBaseProject) return m_pBaseProject->GetNetworkModel();

	return pModel;
}

int CKDProjectManager::GetMinibatchSize() const {
	int size = m_pTrainMode ? m_pTrainMode->GetMinibatchSize() : 0;
	if (!size && m_pBaseProject) return m_pBaseProject->GetMinibatchSize();

	return size;
}

enum enumMinibatchType CKDProjectManager::GetMinibatchType() const {
	enum enumMinibatchType type = m_pTrainMode ? m_pTrainMode->GetMinibatchType() : minibatch_undef;
	if (type == minibatch_undef && m_pBaseProject) return m_pBaseProject->GetMinibatchType();

	return type;
}

int CKDProjectManager::GetTrainExampleCount() const {
	int count = m_pDataSet ? m_pDataSet->GetTrainExampleCount() : 0;
	if (!count && m_pBaseProject) return m_pBaseProject->GetTrainExampleCount();

	return count;
}

int CKDProjectManager::GetTestExampleCount(bool testset) const {
	if (!testset) return GetTrainExampleCount();

	int count = m_pDataSet ? m_pDataSet->GetTestExampleCount() : 0;
	if (!count && m_pBaseProject) return m_pBaseProject->GetTestExampleCount(testset);

	return count;
}

int CKDProjectManager::GetTestCount(bool testset) const {
	int count = m_pTrainMode ? m_pTrainMode->GetTestCount(testset) : 0;
	if (count <= 0 && m_pBaseProject) return m_pBaseProject->GetTestCount(testset);

	return count;
}

bool CKDProjectManager::GetNthTrainExample(int nth, CKDTensor& input, CKDTensor& output) const {
	bool isok = m_pDataSet ? m_pDataSet->GetNthTrainExample(nth, input, output) : false;
	if (!isok && m_pBaseProject) isok = m_pBaseProject->GetNthTrainExample(nth, input, output);
	return isok;
}

bool CKDProjectManager::GetNthTestExample(bool testset, int nth, CKDTensor& input, CKDTensor& output) const {
	if (!testset) return GetNthTrainExample(nth, input, output);

	bool isok = m_pDataSet ? m_pDataSet->GetNthTestExample(nth, input, output) : false;
	if (!isok && m_pBaseProject) isok = m_pBaseProject->GetNthTestExample(testset, nth, input, output);
	return isok;
}

/*
int CKDProjectManager::GetTestCountForTrain() const {
	int count = m_pTrainMode ? m_pTrainMode->GetTestCountForTrain() : 0;
	if (!count && m_pBaseProject) return m_pBaseProject->GetTestCountForTrain();

	return count;
}

int CKDProjectManager::GetTestCountForTest() const {
	int count = m_pTrainMode ? m_pTrainMode->GetTestCountForTest() : 0;
	if (!count && m_pBaseProject) return m_pBaseProject->GetTestCountForTest();

	return count;
}

bool CKDProjectManager::GetNthTrainExample(int nth, CKDTensor& input, CKDTensor& output) const {
	bool isok = m_pDataSet ? m_pDataSet->GetNthTrainExample(nth, input, output) : false;
	if (!isok && m_pBaseProject) isok = m_pBaseProject->GetNthTrainExample(nth, input, output);
	return isok;
}

bool CKDProjectManager::GetNthTestExample(int nth, CKDTensor& input, CKDTensor& output) const {
	bool isok = m_pDataSet ? m_pDataSet->GetNthTestExample(nth, input, output) : false;
	if (!isok && m_pBaseProject) isok = m_pBaseProject->GetNthTestExample(nth, input, output);
	return isok;
}

bool CKDProjectManager::GradeTestResult(CKDTensor estimate, CKDTensor output) {
	if (m_projectMode == proj_mode_regression) {
		return (estimate - output).all_near_zero();
	}
	else if (m_projectMode == proj_mode_binary) {
		if (estimate.size() != 1) THROW_TEMP;
		if (output.size() != 1) THROW_TEMP;

		if (output[0] > 0.5) return estimate[0] > 0.99;
		else return estimate[0] < 0.01;
	}
	else if (m_projectMode == proj_mode_classify) {
		if (estimate.size() != output.size()) THROW_TEMP;

		for (int i = 0; i < estimate.size(); i++) {
			int nEstIndex = estimate.get_max_index();
			int nAnsIndex = output.get_max_index();

			return nEstIndex == nAnsIndex;
		}

		return true;
	}
	else {
		THROW_TEMP;
	}
}
*/

void CKDProjectManager::EvalRegularCost(const CKDTensor &estimate, const CKDTensor &answer, CKDLearningEngine* pEngine, double* pCost) const {
	if (m_pTrainMode) m_pTrainMode->EvalRegularCost(estimate, answer, pEngine, pCost);
	if (m_pBaseProject) m_pBaseProject->EvalRegularCost(estimate, answer, pEngine, pCost);
}

void CKDProjectManager::EvalRegularGrad(const CKDTensor &estimate, const CKDTensor &answer, CKDLearningEngine* pEngine, CKDTensor* pGrad) const {
	if (m_pTrainMode) m_pTrainMode->EvalRegularGrad(estimate, answer, pEngine, pGrad);
	if (m_pBaseProject) m_pBaseProject->EvalRegularGrad(estimate, answer, pEngine, pGrad);
}

double CKDProjectManager::GetWeightDecayNorm2Ratio() const {
	double ratio = m_pTrainMode ? m_pTrainMode->GetWeightDecayNorm2Ratio() : 0;
	if (!ratio && m_pBaseProject) m_pBaseProject->GetWeightDecayNorm2Ratio();
	return ratio;
}

CKDTestValidateInfo* CKDProjectManager::GetValidateSettings() {
	CKDTestValidateInfo* pValidate = m_pTrainMode ? m_pTrainMode->GetValidateSettings() : NULL;
	if (!pValidate && m_pBaseProject) pValidate = m_pBaseProject->GetValidateSettings();
	return pValidate;
}
