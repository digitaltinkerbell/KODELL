#pragma once

#include "../math/tensor.h"
#include "../util/util.h"
#include "../project/network_model.h"

#define DEF_PARAM_SMALL_POS	1.0e-6

class CKDContextStack;
class CKDReporterPool;
class CKDProjectManager;

struct CKDLayerConfig;
struct CKDParameterInitValues;

enum costFuncType;

class CKDLayer {
public:
	CKDLayer();
	virtual ~CKDLayer();

	static CKDLayer* Create(const CKDLayerConfig* pLayerConfig, CKDDimension inputDim, CKDDimension outputDim);

	virtual CKDDimension GetOutputDimension() const = 0;

	//void InitParameter(const CKDParameterInitValues* pInit);
	virtual void ExecFowardStep(CKDContextStack& context, CKDReporterPool* pReporters) = 0; // bool bNeebCack, const CKDTensor input, const CKDTensor &output, CKDTensor &net, CKDReporterPool* pReporters) = 0;
	virtual void ExecBackwardStep(CKDContextStack& context, CKDReporterPool* pReporters) = 0; // const CKDTensor gradient, CKDReporterPool* pReporters) = NULL;
	
	virtual void ResetBackGradient() = 0;
	virtual bool FeedGradient(CKDContextStack& context, double rate, int batch_size, CKDReporterPool* pReporters, CKDProjectManager* pConfig) = 0;
	virtual void SaveFinalResult(CKDReporterPool* pReporters) = 0;

	virtual double GetWeightSqSum() const = 0;
	virtual double GetWeightAbsSum() const = 0;

	virtual enum costFuncType GetCostFunc() const { THROW_TEMP; }

protected:
	void m_execFowardStep(CKDContextStack& context, CKDReporterPool* pReporters);
	void m_execBackwardStep(CKDContextStack& context, CKDReporterPool* pReporters);

	virtual CKDTensor m_execLenearTransform(CKDContextStack& context, CKDTensor input);
	virtual CKDTensor m_execAddBias(CKDContextStack& context, CKDTensor linear);
	virtual CKDTensor m_execActivateFunc(CKDContextStack& context, CKDTensor affine);
	virtual CKDTensor m_execDeactivateFunc(CKDContextStack& context, CKDTensor gradient, CKDTensor out);

	virtual enum biasType m_get_bias_mode() { THROW_TEMP; }
	virtual CKDTensor m_get_bias_vector() { THROW_TEMP; }
	virtual double m_get_bias_value() { THROW_TEMP; }

	virtual enum actFuncType m_get_activate_func() { THROW_TEMP; }

	virtual void m_evalWeightGradient(CKDContextStack& context, CKDTensor grad) { THROW_TEMP; }
	virtual void m_evalBiasGradient(CKDContextStack& context, CKDTensor grad) { THROW_TEMP; }
	virtual CKDTensor m_evalInputGradient(CKDContextStack& context, CKDTensor grad) { THROW_TEMP; }

};

