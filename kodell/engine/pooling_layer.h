#pragma once

#include "../engine/layer.h"

class CKDContextStack;

class CKDPoolingLayer : public CKDLayer {
public:
	CKDPoolingLayer(const CKDPoolingLayerConfig* pLayerConfig, CKDDimension inputDim);
	virtual ~CKDPoolingLayer();

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
	CKDTensor m_execPoolingFunc(CKDContextStack& context, CKDTensor input);
	enum actFuncType m_get_activate_func() { return m_pLayerConfig->m_actFunc; }
	CKDTensor m_evalInputGradient(CKDContextStack& context, CKDTensor grad);

protected:
	const CKDPoolingLayerConfig* m_pLayerConfig;

	CKDDimension m_inputDimension;
	CKDDimension m_outputDimension;

	int* m_router;
	CKDTensor m_jacobian;
};
