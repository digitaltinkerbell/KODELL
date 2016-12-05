#pragma once

#include "../engine/layer.h"

class CKDFullConnectionLayer : public CKDLayer {
public:
	CKDFullConnectionLayer(const CKDFullLayerConfig* pLayerConfig, CKDDimension inputDim, CKDDimension outputDim);
	virtual ~CKDFullConnectionLayer();

	CKDDimension GetOutputDimension() const;

	//void InitParameter(const CKDParameterInitValues* pInit);
	//CKDTensor ExecFowardStep(bool bNeebCack, const CKDTensor input, const CKDTensor &output, CKDTensor &net, CKDReporterPool* pReporters);
	//CKDTensor ExecBackwardStep(const CKDTensor gradient, CKDReporterPool* pReporters);
	void ExecFowardStep(CKDContextStack& context, CKDReporterPool* pReporters);
	void ExecBackwardStep(CKDContextStack& context, CKDReporterPool* pReporters);

	void ResetBackGradient();
	bool FeedGradient(CKDContextStack& context, double rate, int batch_size, CKDReporterPool* pReporters, CKDProjectManager* pConfig);
	void SaveFinalResult(CKDReporterPool* pReporters);

	double GetWeightSqSum() const;
	double GetWeightAbsSum() const;

protected:
	CKDTensor m_execLenearTransform(CKDContextStack& context, CKDTensor input);
	
	void m_evalWeightGradient(CKDContextStack& context, CKDTensor grad);
	void m_evalBiasGradient(CKDContextStack& context, CKDTensor grad);
	
	CKDTensor m_evalInputGradient(CKDContextStack& context, CKDTensor grad);
	
	enum biasType m_get_bias_mode() { return m_pLayerConfig->m_bias; }
	enum actFuncType m_get_activate_func() { return m_pLayerConfig->m_actFunc; }

	CKDTensor m_get_bias_vector() { return m_bias; }
	double m_get_bias_value() { return m_bias[0]; }

protected:
	const CKDFullLayerConfig* m_pLayerConfig;

	CKDTensor m_weight;
	CKDTensor m_bias;

	/*
	CKDTensor m_input;
	CKDTensor m_jacobian;
	//Vector m_grad;

	CKDTensor m_weight_grad;
	CKDTensor m_bias_grad;
	double m_bias_diff;

	double m_weight_diff_sq_sum;
	double m_bias_diff_sq_sum;
	*/
};

class CKDOutputLayer : public CKDFullConnectionLayer {
public:
	CKDOutputLayer(const CKDOutputLayerConfig* pLayerConfig, CKDDimension inputDim, CKDDimension outputDim);
	virtual ~CKDOutputLayer() {}

	virtual enum costFuncType GetCostFunc() const { return m_costFunc; }
protected:
	enum costFuncType m_costFunc;
};

class CKDHiddenLayer : public CKDFullConnectionLayer {
public:
	CKDHiddenLayer(const CKDHiddenLayerConfig* pLayerConfig, CKDDimension inputDim);
	virtual ~CKDHiddenLayer() {}

protected:
};
