#pragma once

#include "../config/types.h"
#include "../util/util.h"

struct CKDParameterInitValues {
	CKDParameterInitValues() { m_isvalid = 0;  }
	int m_isvalid;
	CKDNumbers weights;
	CKDNumbers bias;
};

struct CKDLayerConfig {
	enum enumLayerType m_layerType;
	virtual int GetDepth() const { return 1; }
};

struct CKDFullLayerConfig : public CKDLayerConfig {
	enum actFuncType m_actFunc;
	enum biasType m_bias;
	int m_width;

	CKDParameterInitValues m_inintVlaues;
};

struct CKDOutputLayerConfig : public CKDFullLayerConfig {
	enum costFuncType m_costFunc;
};

struct CKDHiddenLayerConfig : public CKDFullLayerConfig {
	int m_depth;
	int GetDepth() const { return m_depth; }
};

struct CKDConvolutionLayerConfig : public CKDLayerConfig {
	enum actFuncType m_actFunc;
	enum biasType m_bias;
	int m_width;
	int m_height;
	int m_channels;
};

struct CKDPoolingLayerConfig : public CKDLayerConfig {
	enum poolingFuncType m_poolFunc;
	enum actFuncType m_actFunc;
	int m_stride_x;
	int m_stride_y;
};


struct CKDRecurrentLayerConfig : public CKDLayerConfig {
	enum actFuncType m_actFunc;
	enum biasType m_bias;
	int m_width;
};

class CKDNetworkModel {
public:
	CKDNetworkModel();
	virtual ~CKDNetworkModel();

	static CKDNetworkModel* Create(XMLNode* node);

	void Setup(XMLNode* node);

	int GetLayerInfoCount() const { return m_nLayerCount; }
	int GetActualLayerCount() const;

	const CKDLayerConfig* GetNthLayerInfo(int index) const;
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

protected:
	int m_nLayerCount;
	CKDLayerConfig** m_ppLayerConfigs;

	/*
	const char* m_hiddenFunc;
	const char* m_hiddenBias;
	
	int m_hiddenDepth;
	int m_hiddenWidth;

	CKDParameterInitValues m_hiddenInit;
	*/

	void m_setInitValues(CKDParameterInitValues* pBuffer, XMLNode* node);
};
