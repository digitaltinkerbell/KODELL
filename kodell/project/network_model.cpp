#include "stdafx.h"

#include "../project/network_model.h"
#include "../util/util.h"
#include "../util/xml_util.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

CKDNetworkModel::CKDNetworkModel() {
	m_nLayerCount = 0;
	m_ppLayerConfigs = NULL;
}

CKDNetworkModel* CKDNetworkModel::Create(XMLNode* node) {
	XMLNode* network_node = CKDXmlUtil::SeekDefClause(node, "network_model");

	if (!network_node) return NULL;

	CKDNetworkModel* pModel = new CKDNetworkModel;
	pModel->Setup(network_node);

	return pModel;
}

void CKDNetworkModel::Setup(XMLNode* network_node) {
	static const char* layer_names[] = { "output", "full", "convolution", "pooling", "recurrent", NULL };
	static const char* cost_func_names[] = { "mse", "cross_entropy", NULL };
	static const char* act_func_names[] = { "bypass", "sigmoid", "tanh", "relu", "softmax", NULL };
	static const char* pooling_func_names[] = { "max", NULL };
	static const char* bias_names[] = { "off", "on", "each", "common", NULL };

	XMLNode* layers_node = CKDXmlUtil::SeekTagChild(network_node, "layers");

	if (layers_node) {
		int count = CKDXmlUtil::GetChildCount(layers_node);

		m_ppLayerConfigs = new CKDLayerConfig*[count];
		memset(m_ppLayerConfigs, 0, sizeof(CKDLayerConfig*)*count);

		for (int i = 0; i < count; i++) {
			XMLNode* child_node = CKDXmlUtil::GetNthChild(layers_node, i);

			enum enumLayerType type = (enum enumLayerType) CKDXmlUtil::GetEnumTag(child_node, layer_names);

			switch (type) {
			case layer_output:
			{
								 CKDOutputLayerConfig* pConfig = new CKDOutputLayerConfig;
								 pConfig->m_layerType = type;
								 pConfig->m_costFunc = (enum costFuncType) CKDXmlUtil::GetEnumAttr(child_node, "cost_func", cost_func_names);
								 pConfig->m_actFunc = (enum actFuncType) CKDXmlUtil::GetEnumAttr(child_node, "act_func", act_func_names);
								 pConfig->m_bias = (enum biasType) CKDXmlUtil::GetEnumAttr(child_node, "bias", bias_names);
								 m_setInitValues(&pConfig->m_inintVlaues, CKDXmlUtil::SeekTagChild(child_node, "init"));
								 m_ppLayerConfigs[m_nLayerCount++] = pConfig;
			}
				break;
			case layer_full:
			{
							   CKDHiddenLayerConfig* pConfig = new CKDHiddenLayerConfig;
							   pConfig->m_layerType = type;
							   pConfig->m_actFunc = (enum actFuncType) CKDXmlUtil::GetEnumAttr(child_node, "act_func", act_func_names);
							   pConfig->m_bias = (enum biasType) CKDXmlUtil::GetEnumAttr(child_node, "bias", bias_names);
							   pConfig->m_depth = (enum biasType) CKDXmlUtil::GetIntAttr(child_node, "depth");
							   pConfig->m_width = (enum biasType) CKDXmlUtil::GetIntAttr(child_node, "width");
							   m_setInitValues(&pConfig->m_inintVlaues, CKDXmlUtil::SeekTagChild(child_node, "init"));
							   m_ppLayerConfigs[m_nLayerCount++] = pConfig;
			}
				break;
			case layer_convolution:
			{
									  CKDConvolutionLayerConfig* pConfig = new CKDConvolutionLayerConfig;
									  pConfig->m_layerType = type;
									  pConfig->m_actFunc = (enum actFuncType) CKDXmlUtil::GetEnumAttr(child_node, "act_func", act_func_names);
									  pConfig->m_bias = (enum biasType) CKDXmlUtil::GetEnumAttr(child_node, "bias", bias_names);
									  pConfig->m_width = (enum biasType) CKDXmlUtil::GetIntAttr(child_node, "width");
									  pConfig->m_height = (enum biasType) CKDXmlUtil::GetIntAttr(child_node, "height");
									  pConfig->m_channels = (enum biasType) CKDXmlUtil::GetIntAttr(child_node, "channels");
									  m_ppLayerConfigs[m_nLayerCount++] = pConfig;
			}
				break;
			case layer_pooling:
			{
								  CKDPoolingLayerConfig* pConfig = new CKDPoolingLayerConfig;
								  pConfig->m_layerType = type;
								  pConfig->m_poolFunc = (enum poolingFuncType) CKDXmlUtil::GetEnumAttr(child_node, "func", pooling_func_names);
								  pConfig->m_actFunc = (enum actFuncType) CKDXmlUtil::GetEnumAttr(child_node, "act_func", act_func_names);
								  pConfig->m_stride_x = (enum biasType) CKDXmlUtil::GetIntAttr(child_node, "stride");
								  pConfig->m_stride_y = pConfig->m_stride_x;
								  m_ppLayerConfigs[m_nLayerCount++] = pConfig;
			}
				break;
			case layer_recurrent:
			{
								  CKDRecurrentLayerConfig* pConfig = new CKDRecurrentLayerConfig;
								  pConfig->m_layerType = type;
								  pConfig->m_actFunc = (enum actFuncType) CKDXmlUtil::GetEnumAttr(child_node, "act_func", act_func_names);
								  pConfig->m_bias = (enum biasType) CKDXmlUtil::GetEnumAttr(child_node, "bias", bias_names);
								  pConfig->m_width = CKDXmlUtil::GetIntAttr(child_node, "width");
								  m_ppLayerConfigs[m_nLayerCount++] = pConfig;
			}
				break;
			default:
				THROW_TEMP;
			}
		}
	}
}

CKDNetworkModel::~CKDNetworkModel() {
	for (int i = 0; i < m_nLayerCount; i++) {
		delete m_ppLayerConfigs[i];
	}
	
	delete m_ppLayerConfigs;
}

int CKDNetworkModel::GetActualLayerCount() const {
	int count = 0;

	for (int i = 0; i < m_nLayerCount; i++) {
		count += m_ppLayerConfigs[i]->GetDepth();
	}

	return count;
}

const CKDLayerConfig* CKDNetworkModel::GetNthLayerInfo(int index) const {
	if (index < 0 || index >= m_nLayerCount) THROW_TEMP;
	return m_ppLayerConfigs[index];
}


void CKDNetworkModel::m_setInitValues(CKDParameterInitValues* pBuffer, XMLNode* node) {
	if (!pBuffer || !node) return;

	pBuffer->m_isvalid = 1;
	pBuffer->weights.SetData(CKDXmlUtil::SeekTagChild(node, "weight"));
	pBuffer->bias.SetData(CKDXmlUtil::SeekTagChild(node, "bias"));
}

/*
int GetHiddenDepth() const { return m_hiddenDepth; }
const char* GetOutputBias() const { return m_outputBias; }
const char* GetOutputActivateFunc() const { return m_outputFunc; }
const char* GetOutputCostFunc() const { return m_costFunc; }

int GetHiddenWidth(int nLayer) const;
const char* GetHiddenActivateFunc(int nLayer) const;
const char* GetHiddenBias(int nLayer) const;

const CKDParameterInitValues* GetLayerInitValue(int nLayer) const;
*/

/*
int CKDNetworkModel::GetHiddenWidth(int nLayer) const {
	if (m_hiddenDepth == 0) return 0;
	if (nLayer < 0 || nLayer >= m_hiddenDepth) THROW_TEMP;
	return m_hiddenWidth;
}

const char* CKDNetworkModel::GetHiddenActivateFunc(int nLayer) const {
	if (m_hiddenFunc == NULL) return NULL;
	if (nLayer < 0 || nLayer >= m_hiddenDepth) THROW_TEMP;
	return m_hiddenFunc;
}

const char* CKDNetworkModel::GetHiddenBias(int nLayer) const { 
	if (m_hiddenBias == NULL) return NULL;
	if (nLayer < 0 || nLayer >= m_hiddenDepth) THROW_TEMP;
	return m_hiddenBias;
}

const CKDParameterInitValues* CKDNetworkModel::GetLayerInitValue(int nLayer) const {
	if (nLayer == m_hiddenDepth) {							// output layer
		if (!m_outputInit.m_isvalid) return NULL;
		return &m_outputInit;
	}

	if (!m_outputInit.m_isvalid) return NULL;
	if (nLayer < 0 || nLayer >= m_hiddenDepth) THROW_TEMP;		// hidden layer

	return &m_hiddenInit;
}
*/