#pragma once

#include "../math/dimension.h"

#define EPSSILON 1.0e-12

inline bool near_zero(double value) { return value <= EPSSILON && value >= -EPSSILON; }

class CKDTensorCore {
	friend class CKDTensor;

protected:
	CKDTensorCore() { m_nRefCount = 1;  m_pData = NULL; m_pName = NULL; }
	~CKDTensorCore() { delete[] m_pData; }
	int m_nRefCount;
	double *m_pData;
	const char* m_pName;
};

class CKDTensor {
public:
	CKDTensor();
	CKDTensor(const CKDDimension dimension);
	CKDTensor(const CKDTensor &src);
	CKDTensor(double value);
	virtual ~CKDTensor();

	void setDebugCoreName(const char* pName) { m_pCore->m_pName = pName; }
	CKDTensor Clone() const;
	CKDTensor* extractPtr();

	CKDDimension get_dimension() const;

	void SetScalar(double value);

	void Reset() { SetValue(0); }
	void SetDimension(CKDDimension dimension);
	void ChangeDimension(CKDDimension dimension);
	void SetSize();
	void SetSize(int size);
	void SetSize(int size1, int size2);
	void SetSize(int size1, int size2, int size3);
	void SetValue(double val);
	void SetData(double *pval, int nSize=-1);
	void SetDataToDouble(unsigned char* p, int nSize = -1);
	void InitData(double minVal, double maxVal);

	int size() const { return m_dimension.getSize(); }
	int size(int nDim) const;

	double sum() const;
	double l1_norm() const;
	double l2_norm_sq() const;

	bool is_empty() const;
	bool all_near_zero() const;

	void dump(const char* caption) const;
	void dump(FILE* fid, const char* caption) const;

	double& operator [](int index);
	double operator [](int index) const;

	//CKDVector* ToVector(bool bThrow = true);
	//CKDMatrix* ToMatrix(bool bThrow = true);

	CKDTensorCore* m_pCore;
	CKDDimension m_dimension;

	void m_copy(const CKDTensor& src);

	//double m_get_nth_value(int index) const;
	//double& m_get_nth_ref(int index);

	//void m_set_nth_value(int index, double value);

	const CKDTensor& operator = (const CKDTensor& src);
	const CKDTensor& operator += (double value);
	const CKDTensor& operator += (const CKDTensor& src);
	const CKDTensor& operator -= (double value);
	const CKDTensor& operator -= (const CKDTensor& src);
	const CKDTensor& operator *= (double value);
	const CKDTensor& operator /= (double value);

	CKDTensor operator + (const CKDTensor& src) const;
	CKDTensor operator + (double value);
	CKDTensor operator - (const CKDTensor& src) const;
	CKDTensor operator - (double value) const;
	CKDTensor operator * (const CKDTensor& src) const;
	CKDTensor operator * (double value) const;
	CKDTensor operator / (double value) const;

	void m_sigmoid();
	void m_sigmoid_diff();
	void m_tanh();
	void m_tanh_diff();
	void m_relu();
	void m_relu_diff();
	void m_softmax();
	void m_square();
	//void m_softmax_diff(const CKDTensor& src);

	CKDTensor sigmoid() const;
	CKDTensor sigmoid_diff() const;
	CKDTensor tanh() const;
	CKDTensor tanh_diff() const;
	CKDTensor relu() const;
	CKDTensor relu_diff() const;
	CKDTensor softmax() const;
	CKDTensor square() const;
	CKDTensor fill(double value) const;
	CKDTensor mult_each(const CKDTensor vector) const;
	CKDTensor channel_sum() const;
	//void m_softmax_diff(const CKDTensor& src);

	CKDTensor transpose_mult(const CKDTensor& vector) const;
	CKDTensor softmax_diff(const CKDTensor& answer) const;
	CKDTensor mult_cross(const CKDTensor vector) const;
	CKDTensor to_diagonal() const;
	CKDTensor convolution(const CKDTensor& input) const;
	CKDTensor inv_conv_weight(const CKDTensor term, CKDDimension conv_dim) const;
	CKDTensor inv_conv_input(const CKDTensor term) const;
	CKDTensor concat_to_vector(const CKDTensor term) const;
	CKDTensor pooling_max(int sx, int sy, int* router) const;
	CKDTensor nthComponent(int nth) const;

	void route_gradient(const CKDTensor grad, int* router);

	int get_max_index() const;

	double get(int row, int col) const;
	double get(int row, int col, int channel) const;

	double get_fill0(int row, int col) const;
	double get_fill0(int row, int col, int channel) const;

	void set(int row, int col, double value);
	void set(int row, int col, int channel, double value);
};

/*
class CKDScalar : public CKDTensor {
public:
	CKDScalar();
	//CKDScalar(const CKDMatrix& src);
	virtual ~CKDScalar();
};

class CKDVector : public CKDTensor {
public:
	CKDVector();
	CKDVector(int size);
	CKDVector(CKDDimension dimension);
	CKDVector(const CKDVector& src);
	virtual ~CKDVector();

	CKDMatrix softmax_diff(const CKDVector vector, const CKDVector& answer) const;

	//CKDVector square() const;
	//CKDVector fill(double value) const;

	CKDMatrix mult_cross(const CKDVector vector) const;
	CKDMatrix to_diagonal() const;

protected:
};

class CKDMatrix : public CKDTensor {
public:
	CKDMatrix();
	CKDMatrix(CKDDimension dimension);
	CKDMatrix(const CKDMatrix& src);
	virtual ~CKDMatrix();

	const CKDMatrix& operator = (const CKDMatrix& src);
	const CKDMatrix& operator += (const CKDMatrix& src);

	CKDMatrix operator * (const CKDMatrix src) const;
	CKDVector operator * (const CKDVector src) const;

	CKDMatrix operator - (const CKDMatrix src) const;
	CKDMatrix operator * (double value) const;
	CKDMatrix operator / (double value) const;

	int getRows() const { return m_dimension.getSize(0); }
	int getCols() const { return m_dimension.getSize(1); }

	double get(int row, int col) const;
	void set(int row, int col, double value);

protected:
};

class CKDCube : public CKDTensor {
public:
	CKDCube();
	CKDCube(int channels, int rows, int cols);
	CKDCube(const CKDCube& src);
	virtual ~CKDCube();

	void SetSize(int channels, int rows, int cols);

protected:
};
*/

/*
class CKDTempTensorXXX {
public:
	CKDTempTensorXXX();
	CKDTempTensorXXX(const CKDDimension& dim);
	CKDTempTensorXXX(const CKDTempTensorXXX& src);
	CKDTempTensorXXX(CKDTensor& src);

	virtual ~CKDTempTensorXXX();

	void SetDimension (CKDDimension dimension);

	CKDDimension get_dimension() const;

	void dump(const char* caption) const;
	void dump(FILE* fid, const char* caption) const;

	void SetSize();
	void SetSize(int size);
	void SetSize(int size1, int size2);

	void Reset();
	void SetValue(double val);
	void SetData(double *pval, int nSize = -1);
	void SetDataToDouble(unsigned char* p, int nSize = -1);
	void InitData(double minVal, double maxVal);

	double sum() const;
	double l1_norm() const;
	double l2_norm_sq() const;

	bool all_near_zero() const;

	int size() const;
	int size(int nDim) const;
	int get_max_index() const;

	CKDTempTensorXXX transpose_mult(const CKDTempTensorXXX& vector) const;

	const CKDTempTensorXXX& operator = (const CKDTempTensorXXX& src);
	const CKDTempTensorXXX& operator += (const CKDTempTensorXXX& src);

	CKDTempTensorXXX operator + (const CKDTempTensorXXX& src) const;
	CKDTempTensorXXX operator + (double value);
	CKDTempTensorXXX operator - (const CKDTempTensorXXX& src) const;
	CKDTempTensorXXX operator - (double value) const;
	CKDTempTensorXXX operator / (double value) const;
	CKDTempTensorXXX operator * (double value) const;
	CKDTempTensorXXX operator * (CKDTempTensorXXX value) const;

	double& operator [](int index);
	double operator [](int index) const;

	CKDTempTensorXXX sigmoid() const;
	CKDTempTensorXXX sigmoid_diff() const;
	CKDTempTensorXXX relu() const;
	CKDTempTensorXXX relu_diff() const;
	CKDTempTensorXXX softmax() const;

	CKDTempTensorXXX softmax_diff(const CKDTempTensorXXX vector, const CKDTempTensorXXX& answer) const;

	CKDTempTensorXXX square() const;
	CKDTempTensorXXX fill(double value) const;
	CKDTempTensorXXX mult_each(const CKDTempTensorXXX vector) const;

	CKDTempTensorXXX mult_cross(const CKDTempTensorXXX vector) const;
	CKDTempTensorXXX to_diagonal() const;

protected:
	CKDTensor* m_pTensor;
};
*/