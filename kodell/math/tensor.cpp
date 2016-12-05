#include "stdafx.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../math/tensor.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

//static int nth = 0;

CKDTensor::CKDTensor() {
	m_pCore = NULL;
}

CKDTensor::CKDTensor(const CKDDimension dimension) {
	m_dimension = dimension;
	m_pCore = new CKDTensorCore;
	m_pCore->m_pData = new double[m_dimension.getSize()];
	/*
	if (nth == 27) {
		int n = 0;
	}
	TRACE("%d: 0x%08X\n", nth++, m_pCore->m_pData);
	*/
	memset(m_pCore->m_pData, 0, sizeof(double)*m_dimension.getSize());
}

CKDTensor::CKDTensor(const CKDTensor &src) {
	m_dimension = src.m_dimension;
	m_pCore = src.m_pCore;
	if (m_pCore) m_pCore->m_nRefCount++;
}

CKDTensor::CKDTensor(double value) {
	m_dimension = CKDDimension(0, 1);
	m_pCore = new CKDTensorCore;
	m_pCore->m_pData = new double[1];
	m_pCore->m_pData[0] = value;
}

CKDTensor CKDTensor::Clone() const {
	CKDTensor clone(m_dimension);
	//clone.m_pCore = new CKDTensorCore;
	//clone.m_pCore->m_pData = new double[m_dimension.getSize()];
	memcpy(clone.m_pCore->m_pData, m_pCore->m_pData, sizeof(double)*m_dimension.getSize());
	return clone;
}

CKDTensor* CKDTensor::extractPtr() {
	m_pCore->m_nRefCount++;
	return this;
}

CKDTensor::~CKDTensor() {
	if (m_pCore && --m_pCore->m_nRefCount <= 0) {
		delete m_pCore;
	}
}

int CKDTensor::size(int nAxis) const {
	return m_dimension.getAxisSize(nAxis);
}

void CKDTensor::SetSize() {
	SetDimension(CKDDimension(0));
}

void CKDTensor::SetSize(int size) {
	SetDimension(CKDDimension(1, size));
}

void CKDTensor::SetSize(int size1, int size2) {
	SetDimension(CKDDimension(2, size1, size2));
}

void CKDTensor::SetSize(int size1, int size2, int size3) {
	SetDimension(CKDDimension(3, size1, size2, size3));
}

CKDDimension CKDTensor::get_dimension() const {
	return m_dimension;
}

void CKDTensor::SetValue(double val) {
	if (!m_pCore) return;

	double* data = m_pCore->m_pData;

	for (int i = 0; i < m_dimension.getSize(); i++) {
		*data++ = val;
	}
}

void CKDTensor::dump(const char* caption) const {
	dump(stdout, caption);
	return;
	/*
	switch (m_dimension.getDimension()) {
	case 0:
		printf("%s:", caption);
		printf(" %12.9le", (*this)[0]);
		printf("\n");
		break;
	case 1:
		{
			  int size = m_dimension.getSize();
			  printf("%s:", caption);
			  for (int i = 0; i < size; i++) {
				  printf(" %12.9le", (*this)[i]);
			  }
			  printf("\n");
		}
		break;
	case 2:
		{
			  int trows = m_dimension.getAxisSize(0);
			  int tcols = m_dimension.getAxisSize(1);
			  printf("%s:\n\t", caption);
			  for (int i = 0, k = 0; i < trows; i++) {
				  for (int j = 0; j < tcols; j++) {
					  printf(" %12.9lf", (*this)[k++]);
				  }
				  printf("\n\t");
			  }
			  printf("\n");
		}
		break;
	case 3:
		{
			  int tchannels = m_dimension.getAxisSize(0);
			  int trows = m_dimension.getAxisSize(1);
			  int tcols = m_dimension.getAxisSize(2);
			  printf("%s:\n\t", caption);
			  for (int i = 0, m = 0; i < tchannels; i++) {
				  for (int j = 0; j < trows; j++) {
					  for (int k = 0; k < tcols; k++) {
						  printf(" %12.9lf", (*this)[m++]);
					  }
				  }
				  printf("\n\t");
			  }
			  printf("\n");
		}
		break;
	}
	*/
}

void CKDTensor::dump(FILE* fid, const char* caption) const {
	if (!m_pCore) {
		fprintf(fid, "\t%s: empty\n", caption);
		fprintf(fid, "\n");
		return;
	}

	switch (m_dimension.getDimension()) {
	case 0:
		fprintf(fid, "\t%s: scalar = %9.6lf\n", caption, (*this)[0]);
		fprintf(fid, "\n");
		break;
	case 1:
		{
			  int size = m_dimension.getSize();
			  fprintf(fid, "\t%s: vector[%d]:\n", caption, size);
			  for (int i = 0; i < size; i++) {
				  if ((*this)[i] == 0) continue;
				  fprintf(fid, "\t\t%s[%d]=%9.6lf\n", caption, i, (*this)[i]);
			  }
			  fprintf(fid, "\n");
		}
		break;
	case 2:
		{
			  int trows = m_dimension.getAxisSize(0);
			  int tcols = m_dimension.getAxisSize(1);
			  fprintf(fid, "\t%s: maxtrix[%d,%d]:\n", caption, trows, tcols);
			  for (int i = 0, k = 0; i < trows; i++) {
				  for (int j = 0; j < tcols; j++) {
					  double value = (*this)[k++];
					  if (value == 0) continue;
					  fprintf(fid, "\t\t%s[%d,%d]=%9.6lf\n", caption, i, j, value);
				  }
			  }
			  fprintf(fid, "\n");
		}
		break;
	case 3:
		{
			  int tchannels = m_dimension.getAxisSize(0);
			  int trows = m_dimension.getAxisSize(1);
			  int tcols = m_dimension.getAxisSize(2);
			  fprintf(fid, "\t%s: cube[%d,%d, %d]:\n", caption, trows, tcols, tchannels);
			  for (int i = 0, m = 0; i < tchannels; i++) {
				  for (int j = 0; j < trows; j++) {
					  for (int k = 0; k < tcols; k++) {
						  double value = (*this)[m++];
						  if (value == 0) continue;
						  fprintf(fid, "\t\t%s[%d,%d,%d]=%9.6lf\n", caption, i, j, k, value);
					  }
				  }
			  }
			  fprintf(fid, "\n");
		}
		break;
	}
}

/*
CKDVector* CKDTensor::ToVector(bool bThrow) {
	if (m_dimension.getDimension() == 1) return (CKDVector*) this;
	if (bThrow) THROW_TEMP;
	return NULL;
}

CKDMatrix* CKDTensor::ToMatrix(bool bThrow) {
	if (m_dimension.getDimension() == 2) return (CKDMatrix*) this;
	if (bThrow) THROW_TEMP;
	return NULL;
}
*/

CKDTensor CKDTensor::transpose_mult(const CKDTensor& vector) const {
	if (m_dimension.getDimension() == 2) { // && vector.m_dimension.getDimension() == 1) {
		int rows = m_dimension.getAxisSize(0);
		int cols = m_dimension.getAxisSize(1);

		if (rows != vector.size()) THROW_TEMP;

		CKDTensor result(CKDDimension(1, cols));

		for (int i = 0; i < cols; i++) {
			double prod = 0;

			for (int k = 0; k < rows; k++) {
				prod += get(k, i) * vector[k];
			}

			result[i] = prod;
		}

		return result;
	}

	return CKDTensor();
}

CKDTensor CKDTensor::convolution(const CKDTensor& input) const {
	if (m_dimension.getDimension() == input.m_dimension.getDimension() + 1) {
		int nrows = m_dimension.getAxisSize(0);
		int ncols = m_dimension.getAxisSize(1);
		int nchannels = m_dimension.getAxisSize(m_dimension.getDimension() - 1);
		int nrest = m_dimension.getSize() / (nrows * ncols * nchannels);

		int mrows = input.m_dimension.getAxisSize(0);
		int mcols = input.m_dimension.getAxisSize(1);
		int mrest = input.m_dimension.getSize() / (mrows * mcols);

		if (nrest != mrest) THROW_TEMP;

		CKDDimension resultDimension = input.m_dimension.appendAxis(nchannels);
		CKDTensor result(resultDimension);
		const CKDTensor& self = *this;

		for (int nn = 0; nn < nchannels; nn++) {
			for (int mr = 0; mr < mrows; mr++) {
				for (int mc = 0; mc < mcols; mc++) {
					for (int nx = 0; nx < nrest; nx++) {
						double prod = 0;

						for (int nr = 0; nr < nrows; nr++) {
							if (mr + nr >= mrows) continue;
							for (int nc = 0; nc < ncols; nc++) {
								if (mc + nc >= mcols) continue;
								int nidx = (((nr * ncols) + nc) * nrest + nx) * nchannels + nn;
								int midx = (((mr + nr) * mcols) + (mc + nc)) * mrest + nx;
								prod += self[nidx] * input[midx];
							}
						}

						int ridx = (((mr * mcols) + mc) * mrest + nx) * nchannels + nn;
						result[ridx] = prod;
					}
				}
			}
		}

		return result;
	}
	else {
		THROW_TEMP;
	}

	return CKDTensor();
	/*
	if (m_dimension.getDimension() == 3 && input.m_dimension.getDimension() == 2) {
		int cw = m_dimension.getAxisSize(0);
		int ch = m_dimension.getAxisSize(1);
		int channel = m_dimension.getAxisSize(2);

		int iw = input.m_dimension.getAxisSize(0);
		int ih = input.m_dimension.getAxisSize(1);

		CKDTensor result(CKDDimension(3, iw, ih, channel));

		for (int n1 = 0; n1 < channel; n1++) {
			for (int n2 = 0; n2 < iw; n2++) {
				for (int n3 = 0; n3 < ih; n3++) {
					double prod = 0;

					for (int n4 = 0; n4 < cw; n4++) {
						for (int n5 = 0; n5 < ch; n5++) {
							prod += get(n4, n5, n1) * input.get_fill0(n2+n4, n3+n5);
						}
					}

					result.set(n2, n3, n1, prod);
				}
			}
		}

		return result;
	}
	else {
		THROW_TEMP;
	}

	return CKDTensor();
	*/
}

double CKDTensor::get(int row, int col) const {
	if (m_dimension.getDimension() != 2) THROW_TEMP;

	int rows = m_dimension.getAxisSize(0);
	int cols = m_dimension.getAxisSize(1);

	if (row < 0 || row >= rows) THROW_TEMP;
	if (col < 0 || col >= cols) THROW_TEMP;

	return (*this)[row * cols + col];
}

double CKDTensor::get(int row, int col, int channel) const {
	if (m_dimension.getDimension() != 3) THROW_TEMP;

	int rows = m_dimension.getAxisSize(0);
	int cols = m_dimension.getAxisSize(1);
	int channels = m_dimension.getAxisSize(2);

	if (row < 0 || row >= rows) THROW_TEMP;
	if (col < 0 || col >= cols) THROW_TEMP;
	if (channel < 0 || channel >= channels) THROW_TEMP;

	return (*this)[(row * cols + col)*channels + channel];
}

double CKDTensor::get_fill0(int row, int col) const {
	if (m_dimension.getDimension() != 2) THROW_TEMP;

	int rows = m_dimension.getAxisSize(0);
	int cols = m_dimension.getAxisSize(1);

	if (row < 0 || row >= rows) return 0;
	if (col < 0 || col >= cols) return 0;

	return (*this)[row * cols + col];
}

double CKDTensor::get_fill0(int row, int col, int channel) const {
	if (m_dimension.getDimension() != 3) THROW_TEMP;

	int rows = m_dimension.getAxisSize(0);
	int cols = m_dimension.getAxisSize(1);
	int channels = m_dimension.getAxisSize(2);

	if (row < 0 || row >= rows) return 0;
	if (col < 0 || col >= cols) return 0;
	if (channel < 0 || channel >= channels) return 0;

	return (*this)[(row * cols + col)*channels + channel];
}

void CKDTensor::set(int row, int col, double value) {
	if (m_dimension.getDimension() != 2) THROW_TEMP;

	int rows = m_dimension.getAxisSize(0);
	int cols = m_dimension.getAxisSize(1);

	if (row < 0 || row >= rows) THROW_TEMP;
	if (col < 0 || col >= cols) THROW_TEMP;

	(*this)[row * cols + col] = value;
}

void CKDTensor::set(int row, int col, int channel, double value) {
	if (m_dimension.getDimension() != 3) THROW_TEMP;

	int rows = m_dimension.getAxisSize(0);
	int cols = m_dimension.getAxisSize(1);
	int channels = m_dimension.getAxisSize(2);

	if (row < 0 || row >= rows) THROW_TEMP;
	if (col < 0 || col >= cols) THROW_TEMP;
	if (channel < 0 || channel >= channels) THROW_TEMP;

	(*this)[(row * cols + col)*channels + channel] = value;
}

void CKDTensor::SetData(double* pval, int nSize) {
	if (!m_pCore) return;

	int size = m_dimension.getSize();
	if (nSize >= 0 && nSize < size) size = nSize;
	
	memcpy(m_pCore->m_pData, pval, sizeof(double)*size);
}

void CKDTensor::SetDataToDouble(unsigned char* p, int nSize) {
	if (!m_pCore) return;

	int size = m_dimension.getSize();
	if (nSize >= 0 && nSize < size) size = nSize;
	
	for (int i = 0; i < size; i++) {
		if (p[i] != 0) {
			int n = 0;
		}
		m_pCore->m_pData[i] = p[i];
	}

	int n = 0;
}

void CKDTensor::InitData(double minVal, double maxVal) {
	if (!m_pCore) return;

	double diff = maxVal - minVal;
	double* data = m_pCore->m_pData;

	for (int i = 0; i < m_dimension.getSize(); i++) {
		*data++ = minVal + diff * (rand() * 1.0 / RAND_MAX);
	}
}

void CKDTensor::SetScalar(double value) {
	SetDimension(CKDDimension(0, 1));
	SetValue(value);
}

void CKDTensor::SetDimension(CKDDimension dimension) {
	if (m_pCore && --m_pCore->m_nRefCount <= 0) {
		delete m_pCore;
	}

	m_pCore = NULL;
	m_dimension.reset();

	if (dimension.isEmpty()) return;

	m_dimension = dimension;

	m_pCore = new CKDTensorCore;
	m_pCore->m_pData = new double[m_dimension.getSize()];
	memset(m_pCore->m_pData, 0, sizeof(double)*m_dimension.getSize());
}

void CKDTensor::ChangeDimension(CKDDimension dimension) {
	if (m_dimension.getSize() != dimension.getSize()) THROW_TEMP;

	m_dimension = dimension;
}

void CKDTensor::m_copy(const CKDTensor& src) {
	if (m_pCore && --m_pCore->m_nRefCount <= 0) {
		delete m_pCore;
	}

	m_pCore = NULL;
	m_dimension = src.m_dimension;

	m_pCore = src.m_pCore;
	if (m_pCore) m_pCore->m_nRefCount++;

}

double& CKDTensor::operator [](int index) {
	if (index < 0 || index >= m_dimension.getSize()) THROW_TEMP;
	return m_pCore->m_pData[index];
}

double CKDTensor::operator [](int index) const {
	if (index < 0 || index >= m_dimension.getSize()) THROW_TEMP;
	return m_pCore->m_pData[index];
}

CKDTensor CKDTensor::operator * (const CKDTensor& src) const {
	if (m_dimension.getSize() == 1) {
		if ((*this)[0] == 1) return src;
		else return src * (*this)[0];
	}
	else if (src.m_dimension.getSize() == 1) {
		if (src[0] == 1) return *this;
		else return (*this) * src[0];
	}
	else if (m_dimension.getDimension() == 2 && src.m_dimension.getDimension() == 2) {
		int trows = m_dimension.getAxisSize(0);
		int tcols = m_dimension.getAxisSize(1);

		int srows = src.m_dimension.getAxisSize(0);
		int scols = src.m_dimension.getAxisSize(1);

		if (tcols != srows) THROW_TEMP;

		CKDTensor matrix(CKDDimension(2, trows, scols));

		for (int i = 0; i < trows; i++) {
			for (int j = 0; j < scols; j++) {
				double prod = 0;
				for (int k = 0; k < tcols; k++) {
					prod += get(i, k) * src.get(k, j);
				}
				matrix.set(i, j, prod);
			}
		}

		return matrix;
	}
	else if (m_dimension.getDimension() == 2 && src.m_dimension.getDimension() == 1) {
		int trows = m_dimension.getAxisSize(0);
		int tcols = m_dimension.getAxisSize(1);

		if (tcols != src.size()) THROW_TEMP;

		CKDTensor vector(CKDDimension(1, trows));

		for (int i = 0; i < trows; i++) {
			double prod = 0;
			for (int k = 0; k < tcols; k++) {
				prod += get(i, k) * src[k];
			}
			vector[i] = prod;
		}

		return vector;
	}
	else if (m_dimension.getDimension() == 1 && src.m_dimension.getDimension() == 1) {
		int tsize = size();

		if (tsize != src.size()) THROW_TEMP;

		CKDTensor vector(CKDDimension(1, tsize));

		for (int k = 0; k < tsize; k++) {
			vector[k] = (*this)[k] * src[k];
		}

		return vector;
	}
	else {
		THROW_TEMP;
	}

	return CKDTensor();
}

/*
double CKDTensor::m_get_nth_value(int index) const {
	if (index < 0 || index >= m_dimension.getSize()) THROW_TEMP;
	return m_pCore->m_pData[index];
}

double& CKDTensor::m_get_nth_ref(int index) {
	if (index < 0 || index >= m_dimension.getSize()) THROW_TEMP;
	return m_pCore->m_pData[index];
}

void CKDTensor::m_set_nth_value(int index, double value) {
	if (index < 0 || index >= m_dimension.getSize()) THROW_TEMP;
	m_pCore->m_pData[index] = value;
}
*/

void CKDTensor::m_sigmoid() {
	for (int i = 0; i < m_dimension.getSize(); i++) {
		m_pCore->m_pData[i] = 1.0 / (1.0 + exp(-m_pCore->m_pData[i]));
	}
}

void CKDTensor::m_sigmoid_diff() {
	for (int i = 0; i < m_dimension.getSize(); i++) {
		m_pCore->m_pData[i] = m_pCore->m_pData[i] * (1 - m_pCore->m_pData[i]);
	}
}

void CKDTensor::m_tanh() {
	for (int i = 0; i < m_dimension.getSize(); i++) {
		m_pCore->m_pData[i] = 2.0 / (1.0 + exp(-2*m_pCore->m_pData[i])) - 1;
	}
}

void CKDTensor::m_tanh_diff() {
	for (int i = 0; i < m_dimension.getSize(); i++) {
		double sum = exp(m_pCore->m_pData[i]) + exp(-m_pCore->m_pData[i]);
		m_pCore->m_pData[i] = 4.0 / (sum * sum);
	}
}

void CKDTensor::m_relu() {
	for (int i = 0; i < m_dimension.getSize(); i++) {
		if (m_pCore->m_pData[i] < 0) m_pCore->m_pData[i] = 0;
	}
}

void CKDTensor::m_relu_diff() {
	for (int i = 0; i < m_dimension.getSize(); i++) {
		m_pCore->m_pData[i] = m_pCore->m_pData[i] > 0 ? 1 : 0;
	}
}

void CKDTensor::m_softmax() {
	if (m_dimension.getSize() < 1) THROW_TEMP;

	double max_val = m_pCore->m_pData[0];
	for (int i = 1; i < m_dimension.getSize(); i++) {
		if (m_pCore->m_pData[i] > max_val) max_val = m_pCore->m_pData[i];
	}

	double exp_sum = 0;
	for (int i = 0; i < m_dimension.getSize(); i++) {
		exp_sum += exp(m_pCore->m_pData[i] - max_val);
	}

	for (int i = 0; i < m_dimension.getSize(); i++) {
		m_pCore->m_pData[i] = exp(m_pCore->m_pData[i] - max_val) / exp_sum;
	}
}

/*
void CKDTensor::m_softmax_diff(const CKDTensor& src) {
	for (int i = 0; i < m_dimension.getSize(); i++) {
		m_pCore->m_pData[i] = src.m_pCore->m_pData[i] * (1 - src.m_pCore->m_pData[i]);
		//m_pCore->m_pData[i] = src.m_pCore->m_pData[i];
		//m_pCore->m_pData[i] = 1;
	}
}
*/

void CKDTensor::m_square() {
	for (int i = 0; i < m_dimension.getSize(); i++) {
		m_pCore->m_pData[i] = m_pCore->m_pData[i] * m_pCore->m_pData[i];
	}
}

double CKDTensor::sum() const {
	double sum = 0;

	for (int i = 0; i < m_dimension.getSize(); i++) {
		sum += m_pCore->m_pData[i];
	}

	return sum;
}

double CKDTensor::l1_norm() const {
	double sum = 0;

	for (int i = 0; i < m_dimension.getSize(); i++) {
		sum += m_pCore->m_pData[i] > 0 ? m_pCore->m_pData[i] : -m_pCore->m_pData[i];
	}

	return sum;
}

double CKDTensor::l2_norm_sq() const {
	double sum = 0;

	for (int i = 0; i < m_dimension.getSize(); i++) {
		sum += m_pCore->m_pData[i] * m_pCore->m_pData[i];
	}

	return sum;
}

bool CKDTensor::all_near_zero() const {
	for (int i = 0; i < m_dimension.getSize(); i++) {
		if (!near_zero(m_pCore->m_pData[i])) return false;
	}

	return true;
}

bool CKDTensor::is_empty() const {
	return m_pCore == NULL || m_dimension.isEmpty();
}

/*
CKDScalar::CKDScalar() : CKDTensor(CKDDimension(0,1)) {
}

CKDScalar::~CKDScalar() {
}

CKDVector::CKDVector() : CKDTensor(CKDDimension(1, 0)) {
}

CKDVector::CKDVector(int size) : CKDTensor(CKDDimension(1, size)) {
}

CKDVector::CKDVector(const CKDVector& src) : CKDTensor(CKDDimension()) {
	m_copy(src);
}

CKDVector::~CKDVector() {
}
*/

const CKDTensor& CKDTensor::operator =(const CKDTensor& src) {
	if (this == &src) return *this;

	m_copy(src);

	return *this;
}

const CKDTensor& CKDTensor::operator += (double value) {
	for (int i = 0; i < m_dimension.getSize(); i++) {
		(*this)[i] += value;
	}

	return *this;
}

const CKDTensor& CKDTensor::operator += (const CKDTensor& src) {
	if (m_pCore == NULL) {
		SetDimension(src.m_dimension);
	}
	else if (m_dimension != src.m_dimension) THROW_TEMP;

	for (int i = 0; i < m_dimension.getSize(); i++) {
		(*this)[i] += src[i];
	}

	return *this;
}

const CKDTensor& CKDTensor::operator -= (double value) {
	for (int i = 0; i < m_dimension.getSize(); i++) {
		(*this)[i] -= value;
	}

	return *this;
}

const CKDTensor& CKDTensor::operator -= (const CKDTensor& src) {
	if (m_pCore == NULL) {
		SetDimension(src.m_dimension);
	}
	else if (m_dimension != src.m_dimension) THROW_TEMP;

	for (int i = 0; i < m_dimension.getSize(); i++) {
		(*this)[i] -= src[i];
	}

	return *this;
}

const CKDTensor& CKDTensor::operator *= (double value) {
	for (int i = 0; i < m_dimension.getSize(); i++) {
		(*this)[i] *= value;
	}

	return *this;
}

const CKDTensor& CKDTensor::operator /= (double value) {
	for (int i = 0; i < m_dimension.getSize(); i++) {
		(*this)[i] /= value;
	}

	return *this;
}


CKDTensor CKDTensor::operator + (const CKDTensor& src) const {
	CKDTensor vector = Clone();

	if (m_dimension == src.m_dimension) {
		for (int i = 0; i < m_dimension.getSize(); i++) {
			vector[i] += src[i];
		}
	}
	else if (m_dimension.include_tail(src.m_dimension)) {
		int n = m_dimension.getSize() / src.m_dimension.getSize();

		for (int i = 0; i < m_dimension.getSize(); i++) {
			vector[i] += src[i/n];
		}
	}
	else {
		THROW_TEMP;
	}

	return vector;
}

CKDTensor CKDTensor::operator + (double src) {
	CKDTensor vector = Clone();

	for (int i = 0; i < m_dimension.getSize(); i++) {
		vector[i] += src;
	}

	return vector;
}

CKDTensor CKDTensor::operator - (const CKDTensor& src) const {
	//if (m_dimension != src.m_dimension) THROW_TEMP;
	if (size() != src.size()) THROW_TEMP;

	CKDTensor vector = Clone();

	for (int i = 0; i < m_dimension.getSize(); i++) {
		vector[i] -= src[i];
	}

	return vector;
}

CKDTensor CKDTensor::operator - (double src) const {
	CKDTensor vector = Clone();

	for (int i = 0; i < m_dimension.getSize(); i++) {
		vector[i] -= src;
	}

	return vector;
}

CKDTensor CKDTensor::operator / (double src) const {
	if (src == 0) THROW_TEMP;

	CKDTensor vector = Clone();

	for (int i = 0; i < m_dimension.getSize(); i++) {
		vector[i] /= src;
	}

	return vector;
}

CKDTensor CKDTensor::operator * (double src) const {
	CKDTensor vector = Clone();

	for (int i = 0; i < m_dimension.getSize(); i++) {
		vector[i] *= src;
	}

	return vector;
}

CKDTensor CKDTensor::sigmoid() const {
	CKDTensor vector = Clone();
	vector.m_sigmoid();
	return vector;
}

CKDTensor CKDTensor::sigmoid_diff() const {
	CKDTensor vector = Clone();
	vector.m_sigmoid_diff();
	return vector;
}

CKDTensor CKDTensor::tanh() const {
	CKDTensor vector = Clone();
	vector.m_tanh();
	return vector;
}

CKDTensor CKDTensor::tanh_diff() const {
	CKDTensor vector = Clone();
	vector.m_tanh_diff();
	return vector;
}

CKDTensor CKDTensor::relu() const {
	CKDTensor vector = Clone();
	vector.m_relu();
	return vector;
}

CKDTensor CKDTensor::relu_diff() const {
	CKDTensor vector = Clone();
	vector.m_relu_diff();
	return vector;
}

CKDTensor CKDTensor::softmax() const {
	CKDTensor vector = Clone();
	vector.m_softmax();
	return vector;
}

CKDTensor CKDTensor::softmax_diff(const CKDTensor& answer) const {
	if (m_dimension.getDimension() != 1) THROW_TEMP;
	if (answer.m_dimension.getDimension() != 1) THROW_TEMP;
		
	int tsize = m_dimension.getSize();
	int asize = answer.m_dimension.getSize();

	if (tsize != asize) THROW_TEMP;

	CKDTensor matrix(CKDDimension(2, tsize, tsize));

	const CKDTensor& self = *this;

	for (int i = 0; i < tsize; i++) {
		for (int j = 0; j < tsize; j++) {
			matrix.set(i, j, self[i] * (((i == j) ? 1 : 0) - self[j]));
		}
	}

	return matrix;
}

CKDTensor CKDTensor::square() const {
	CKDTensor vector = Clone();
	vector.m_square();
	return vector;
}

CKDTensor CKDTensor::channel_sum() const {
	int nDim = m_dimension.getDimension();
	if (nDim < 2) THROW_TEMP;
	int nChannels = m_dimension.getAxisSize(nDim - 1);
	int nSize = m_dimension.getSize();

	const CKDTensor& self = *this;

	CKDTensor vector(CKDDimension(1, nChannels));

	for (int i = 0; i < nSize; i++) {
		vector[i % nChannels] += self[i];
	}

	return vector;
}

CKDTensor CKDTensor::fill(double value) const {
	CKDTensor vector(m_dimension);
	vector.SetValue(value);
	return vector;
}

CKDTensor CKDTensor::mult_each(const CKDTensor src) const {
	if (m_dimension.getSize() != src.m_dimension.getSize()) THROW_TEMP;

	CKDTensor vector = Clone();

	for (int i = 0; i < m_dimension.getSize(); i++) {
		vector[i] *= src[i];
	}

	return vector;
}

CKDTensor CKDTensor::mult_cross(const CKDTensor vector) const {
	//if (m_dimension.getDimension() != 1) THROW_TEMP;
	//if (vector.m_dimension.getDimension() != 1) THROW_TEMP;

	int tsize = m_dimension.getSize();
	int vsize = vector.m_dimension.getSize();

	CKDTensor matrix(CKDDimension(2, tsize, vsize));

	const CKDTensor& self = *this;

	for (int i = 0; i < tsize; i++) {
		for (int j = 0; j < vsize; j++) {
			matrix.set(i, j, self[i] * vector[j]);
		}
	}

	return matrix;
}

CKDTensor CKDTensor::inv_conv_weight(const CKDTensor term, CKDDimension conv_dim) const {
	int dim = m_dimension.getDimension();

	if (term.m_dimension.getDimension() != dim + 1) THROW_TEMP;

	int nrows = m_dimension.getAxisSize(0);
	int ncols = m_dimension.getAxisSize(1);
	int nrest = m_dimension.getSize() / (nrows * ncols);

	int grows = conv_dim.getAxisSize(0);
	int gcols = conv_dim.getAxisSize(1);
	int channels = conv_dim.getAxisSize(dim);
	int grest = conv_dim.getSize() / (grows * gcols * channels);

	int mrows = term.m_dimension.getAxisSize(0);
	int mcols = term.m_dimension.getAxisSize(1);
	int mchannels = term.m_dimension.getAxisSize(dim);
	int mrest = term.m_dimension.getSize() / (mrows * mcols * mchannels);

	if (mrows != nrows) THROW_TEMP;
	if (mcols != ncols) THROW_TEMP;
	if (mchannels != channels) THROW_TEMP;
	if (grest != nrest) THROW_TEMP;
	if (mrest != nrest) THROW_TEMP;

	CKDTensor grad(conv_dim);

	const CKDTensor& self = *this;

	for (int nn = 0; nn < channels; nn++) {
		for (int gr = 0; gr < grows; gr++) {
			for (int gc = 0; gc < gcols; gc++) {
				for (int gx = 0; gx < grest; gx++) {
					double sum = 0;

					for (int mr = 0; mr < mrows; mr++) {
						if (mr + gr >= mrows) continue;

						int nbase = ((mr * mcols) + 0) * mrest + gx;
						int mbase = ((((mr + gr) * mcols) + gc) * mrest + gx) * channels + nn;

						for (int mc = 0; mc < mcols; mc++) {
							if (mc + gc >= mcols) continue;

							int nidx = nbase + mc * mrest;
							int midx = mbase + mc * mrest * channels;

							sum += term[midx] * self[nidx];
						}
					}

					int gidx = (((gr * gcols) + gc) * grest + gx) * channels + nn;
					grad[gidx] = sum;
				}
			}
		}
	}

	return grad;
}

CKDTensor CKDTensor::concat_to_vector(const CKDTensor term) const {
	int size1 = size();
	int size2 = term.size();
	int n = 0;

	CKDTensor result(CKDDimension(1, size1 + size2));
	const CKDTensor& self = *this;

	for (; n < size1; n++) {
		result[n] = self[n];
	}

	for (int i = 0; i < size2; i++, n++) {
		result[n] = term[i];
	}

	return result;
}

CKDTensor CKDTensor::inv_conv_input(const CKDTensor weight) const {
	int dim = m_dimension.getDimension();

	if (dim < 3) THROW_TEMP;
	if (weight.m_dimension.getDimension() != dim) THROW_TEMP;

	int mrows = m_dimension.getAxisSize(0);
	int mcols = m_dimension.getAxisSize(1);
	int channels = m_dimension.getAxisSize(dim - 1);
	int mrest = m_dimension.getSize() / (mrows * mcols * channels);

	int nrows = weight.m_dimension.getAxisSize(0);
	int ncols = weight.m_dimension.getAxisSize(1);
	int nrest = weight.m_dimension.getSize() / (nrows * ncols * channels);

	if (weight.m_dimension.getAxisSize(dim - 1) != channels) THROW_TEMP;
	if (mrest != nrest) THROW_TEMP;

	CKDTensor delta(m_dimension.popTail());

	const CKDTensor& self = *this;

	for (int mr = 0; mr < mrows; mr++) {
		for (int mc = 0; mc < mcols; mc++) {
			for (int xx = 0; xx < mrest; xx++) {
				double sum = 0;

				for (int nr = 0; nr < nrows; nr++) {
					if (mr < nr) continue;
					for (int nc = 0; nc < ncols; nc++) {
						if (mc < nc) continue;
						int mbase = ((((mr - nr) * mcols) + (mc-nc)) * mrest) * channels;
						int nbase = (((nr * ncols) + nc) * nrest) * channels;
						for (int nn = 0; nn < channels; nn++) {
							sum += self[mbase+nn] * weight[nbase+nn];
						}
					}
				}

				int ridx = ((mr * mcols) + mc) * mrest;

				delta[ridx] = sum;
			}
		}
	}

	return delta;
}

CKDTensor CKDTensor::pooling_max(int sx, int sy, int* router) const {
	if (m_dimension.getDimension() < 2) THROW_TEMP;

	int mrows = m_dimension.getAxisSize(0);
	int mcols = m_dimension.getAxisSize(1);
	int msize = m_dimension.getSize();
	int mrest = msize / (mrows * mcols);

	CKDDimension dimension = m_dimension.clone();

	dimension.setAxisSize(0, (m_dimension.getAxisSize(0) + sx - 1) / sx);
	dimension.setAxisSize(1, (m_dimension.getAxisSize(1) + sy - 1) / sy);

	int nrows = dimension.getAxisSize(0);
	int ncols = dimension.getAxisSize(1);
	int nsize = dimension.getSize();
	int nrest = nsize / (nrows * ncols);

	if (mrest != nrest) THROW_TEMP;

	CKDTensor result(dimension);
	const CKDTensor& self = *this;

	for (int nr = 0; nr < nrows; nr++) {
		for (int nc = 0; nc < ncols; nc++) {
			for (int nx = 0; nx < nrest; nx++) {
				int nidx = ((nr*ncols) + nc)*nrest + nx;
				int midx = ((nr*sx*mcols) + nc*sy)*mrest + nx;
				result[nidx] = self[midx];
				router[nidx] = midx;
			}
		}
	}

	for (int midx = 0; midx < msize; midx++) {
		int mrow = midx / (mcols * mrest);
		int mcol = midx / mrest % mcols;
		int etc = midx % mrest;
		int nrow = mrow / sx;
		int ncol = mcol / sy;
		int nidx = ((nrow*ncols) + ncol)*nrest+etc;
		if (result[nidx] < self[midx]) {
			if (nidx == 20) {
				int n = 0;
			}
			result[nidx] = self[midx];
			router[nidx] = midx;
		}
	}

	return result;
}

void CKDTensor::route_gradient(const CKDTensor grad, int* router) {
	int nsize = grad.size();

	CKDTensor& self = *this;

	for (int n = 0; n < nsize; n++) {
		self[router[n]] = grad[n];
	}

}

CKDTensor CKDTensor::nthComponent(int nth) const {
	CKDDimension dimension = m_dimension.popHead();

	const CKDTensor& self = *this;
	CKDTensor component(dimension);

	int pos = dimension.getSize() * nth;
	memcpy(component.m_pCore->m_pData, m_pCore->m_pData+pos, sizeof(double)*dimension.getSize());

	return component;
}

CKDTensor CKDTensor::to_diagonal() const {
	if (m_dimension.getDimension() != 1) THROW_TEMP;

	int size = m_dimension.getSize();

	CKDTensor matrix(CKDDimension(2, size, size));
	const CKDTensor& self = *this;

	for (int i = 0; i < size; i++) {
		matrix.set(i, i, self[i]);
	}

	return matrix;
}

int CKDTensor::get_max_index() const {
	int size = m_dimension.getSize();

	if (size <= 0) THROW_TEMP;

	int idx = 0;
	double max = (*this)[0];

	for (int i = 1; i < size; i++) {
		double val = (*this)[i];
		if (max < val) {
			idx = i;
			max = val;
		}
	}

	return idx;
}

/*
CKDMatrix::CKDMatrix() : CKDTensor(CKDDimension(2,0)) {
}

CKDMatrix::CKDMatrix(CKDDimension dimension) : CKDTensor(dimension) {
	//SetDimension(m_rows * m_cols);
}

CKDMatrix::CKDMatrix(const CKDMatrix& src) : CKDTensor(src.m_dimension) {
	m_copy(src);
}

CKDMatrix::~CKDMatrix() {
}
*/

/*
const CKDMatrix& CKDMatrix::operator = (const CKDMatrix& src) {
	if (this == &src) return *this;

	m_copy(src);

	return *this;
}

const CKDMatrix& CKDMatrix::operator += (const CKDMatrix& src) {
	if (m_pCore == NULL) {
		SetDimension(src.m_dimension);
	}
	else {
		if (m_dimension != src.m_dimension) THROW_TEMP;
	}

	for (int i = 0; i < m_dimension.getSize(); i++) {
		(*this)[i] += src[i];
	}

	return *this;
}

CKDVector CKDMatrix::transpose_mult(const CKDVector& vector) const {
	int rows = m_dimension.getAxisSize(0);
	int cols = m_dimension.getAxisSize(1);

	if (rows != vector.size()) THROW_TEMP;

	CKDVector result(cols);

	for (int i = 0; i < cols; i++) {
		double prod = 0;

		for (int k = 0; k < rows; k++) {
			prod += get(k,i) * vector[k];
		}

		result[i] = prod;
	}

	return result;
}

double CKDMatrix::get(int row, int col) const {
	int rows = m_dimension.getAxisSize(0);
	int cols = m_dimension.getAxisSize(1);

	if (row < 0 || row >= rows) THROW_TEMP;
	if (col < 0 || col >= cols) THROW_TEMP;

	return (*this)[row * cols + col];
}

void CKDMatrix::set(int row, int col, double value) {
	int rows = m_dimension.getAxisSize(0);
	int cols = m_dimension.getAxisSize(1);

	if (row < 0 || row >= rows) THROW_TEMP;
	if (col < 0 || col >= cols) THROW_TEMP;

	(*this)[row * cols + col] = value;
}
*/

/*
CKDMatrix CKDMatrix::operator - (const CKDMatrix src) const {
	if (m_dimension != src.m_dimension) THROW_TEMP;

	CKDMatrix matrix(*this);

	for (int i = 0; i < m_dimension.getSize(); i++) {
		matrix[i] -= src[i];
	}

	return matrix;
}

CKDMatrix CKDMatrix::operator * (double value) const {
	CKDMatrix matrix = Clone();

	for (int i = 0; i < m_dimension.getSize(); i++) {
		matrix[i] *= value;
	}

	return matrix;
}

CKDMatrix CKDMatrix::operator / (double value) const {
	CKDMatrix matrix = Clone();

	for (int i = 0; i < m_dimension.getSize(); i++) {
		matrix[i] /= value;
	}

	return matrix;
}

CKDMatrix CKDMatrix::operator * (const CKDMatrix src) const {
	int trows = m_dimension.getAxisSize(0);
	int tcols = m_dimension.getAxisSize(1);

	int srows = src.m_dimension.getAxisSize(0);
	int scols = src.m_dimension.getAxisSize(1);

	if (tcols != srows) THROW_TEMP;

	CKDMatrix matrix(CKDDimension(2, trows, scols));

	for (int i = 0; i < trows; i++) {
		for (int j = 0; j < scols; j++) {
			double prod = 0;
			for (int k = 0; k < tcols; k++) {
				prod += get(i, k) * src.get(k, j);
			}
			matrix.set(i, j, prod);
		}
	}

	return matrix;
}

CKDVector CKDMatrix::operator * (const CKDVector src) const {
	int trows = m_dimension.getAxisSize(0);
	int tcols = m_dimension.getAxisSize(1);

	if (tcols != src.size()) THROW_TEMP;

	CKDVector vector(trows);

	for (int i = 0; i < trows; i++) {
		double prod = 0;
		for (int k = 0; k < tcols; k++) {
			prod += get(i, k) * src[k];
		}
		vector[i] = prod;
	}

	return vector;
}

CKDCube::CKDCube() : CKDTensor(CKDDimension(3,0)) {
}

CKDCube::CKDCube(int channels, int rows, int cols) : CKDTensor(CKDDimension(3, channels, rows, cols)) {
}

CKDCube::CKDCube(const CKDCube& src) : CKDTensor(src.m_dimension) {
	m_copy(src);
}

CKDCube::~CKDCube() {
}

void CKDCube::SetSize(int channels, int rows, int cols) {
	CKDDimension dimension(3, channels, rows, cols);
	if (m_dimension == dimension) return;

	SetDimension(dimension);
}
*/

/*
CKDTempTensorXXX::CKDTempTensorXXX() {
	m_pTensor = NULL;
}

CKDTempTensorXXX::CKDTempTensorXXX(const CKDDimension& dim) {
	m_pTensor = new CKDTensor(dim);
}

CKDTempTensorXXX::CKDTempTensorXXX(const CKDTempTensorXXX& src) {
	m_pTensor = new CKDTensor(src.get_dimension());
}

CKDTempTensorXXX::CKDTempTensorXXX(CKDTensor& src) {
	m_pTensor = src.extractPtr();
}

CKDTempTensorXXX::~CKDTempTensorXXX() {
	delete m_pTensor;
}

void CKDTempTensorXXX::SetDimension(CKDDimension dimension) {
	delete m_pTensor;
	m_pTensor = new CKDTensor(dimension);
}

CKDDimension CKDTempTensorXXX::get_dimension() const {
	if (!m_pTensor) THROW_TEMP;
	return m_pTensor->get_dimension();
}

void CKDTempTensorXXX::dump(const char* caption) const {
	if (m_pTensor) m_pTensor->Dump(caption);
}

void CKDTempTensorXXX::dump(FILE* fid, const char* caption) const {
	if (m_pTensor) m_pTensor->Dump(fid, caption);
}


void CKDTempTensorXXX::SetSize() {
	delete m_pTensor;
	m_pTensor = new CKDTensor(CKDDimension(0, 0));

}

void CKDTempTensorXXX::SetSize(int size) {
	delete m_pTensor;
	m_pTensor = new CKDTensor(CKDDimension(1, size));
}

void CKDTempTensorXXX::SetSize(int size1, int size2) {
	delete m_pTensor;
	m_pTensor = new CKDTensor(CKDDimension(2, size1, size2));
}

void CKDTempTensorXXX::Reset() {
	if (m_pTensor) m_pTensor->Reset();
}

void CKDTempTensorXXX::SetValue(double val) {
	if (!m_pTensor) THROW_TEMP;
	if (m_pTensor) m_pTensor->SetValue(val);
}

void CKDTempTensorXXX::SetData(double *pval, int nSize) {
	if (!m_pTensor) THROW_TEMP;
	if (m_pTensor) m_pTensor->SetData(pval, nSize);
}

void CKDTempTensorXXX::SetDataToDouble(unsigned char* p, int nSize) {
	if (!m_pTensor) THROW_TEMP;
	if (m_pTensor) m_pTensor->SetDataToDouble(p, nSize);
}

void CKDTempTensorXXX::InitData(double minVal, double maxVal) {
	if (!m_pTensor) THROW_TEMP;
	if (m_pTensor) m_pTensor->InitData(minVal, maxVal);
}

double CKDTempTensorXXX::sum() const {
	if (!m_pTensor) THROW_TEMP;
	return  m_pTensor->sum();
}

double CKDTempTensorXXX::l1_norm() const {
	if (!m_pTensor) THROW_TEMP;
	return  m_pTensor->l1_norm();
}

double CKDTempTensorXXX::l2_norm_sq() const {
	if (!m_pTensor) THROW_TEMP;
	return  m_pTensor->l2_norm_sq();
}

bool CKDTempTensorXXX::all_near_zero() const {
	if (!m_pTensor) THROW_TEMP;
	return  m_pTensor->all_near_zero();
}

int CKDTempTensorXXX::size() const {
	return m_pTensor ? m_pTensor->m_dimension.getSize() : 0;
}

int CKDTempTensorXXX::size(int nDim) const {
	return m_pTensor ? m_pTensor->m_dimension.getAxisSize(nDim) : 0;
}

int CKDTempTensorXXX::get_max_index() const {
	CKDVector* vector = m_pTensor->ToVector();
	return vector->get_max_index();
}

CKDTempTensorXXX CKDTempTensorXXX::transpose_mult(const CKDTempTensorXXX& vec) const {
	CKDMatrix* matrix = m_pTensor->ToMatrix();
	CKDVector* vector = vec.m_pTensor->ToVector();

	return CKDTempTensorXXX(matrix->transpose_mult(*vector));
}

const CKDTempTensorXXX& CKDTempTensorXXX::operator = (const CKDTempTensorXXX& src) {
	if (this == &src) return *this;
	delete m_pTensor;
	//m_pTensor = src.m_pTensor;
	m_pTensor = src.m_pTensor->Clone(); // new CKDTensor(src.m_pTensor);
	//src.m_pTensor->extractPtr();
	return *this;
}

const CKDTempTensorXXX& CKDTempTensorXXX::operator += (const CKDTempTensorXXX& src) {
	if (m_pTensor == NULL) THROW_TEMP;
	if (src.m_pTensor == NULL) THROW_TEMP;

	(*m_pTensor) += (*src.m_pTensor);

	return *this;
}

CKDTempTensorXXX CKDTempTensorXXX::operator + (const CKDTempTensorXXX& src) const {
	if (m_pTensor == NULL) THROW_TEMP;
	if (src.m_pTensor == NULL) THROW_TEMP;
	if (m_pTensor->m_dimension != src.m_pTensor->m_dimension) THROW_TEMP;

	(*m_pTensor) += (*src.m_pTensor);

	return *this;
}

CKDTempTensorXXX CKDTempTensorXXX::operator + (double src) {
	if (m_pTensor == NULL) THROW_TEMP;
	(*m_pTensor) += src;
	return *this;
}

CKDTempTensorXXX CKDTempTensorXXX::operator - (const CKDTempTensorXXX& src) const {
	if (m_pTensor == NULL) THROW_TEMP;
	if (src.m_pTensor == NULL) THROW_TEMP;
	if (m_pTensor->m_dimension != src.m_pTensor->m_dimension) THROW_TEMP;

	(*m_pTensor) -= (*src.m_pTensor);

	return *this;
}

CKDTempTensorXXX CKDTempTensorXXX::operator - (double src) const {
	if (m_pTensor == NULL) THROW_TEMP;
	(*m_pTensor) -= src;
	return *this;
}

CKDTempTensorXXX CKDTempTensorXXX::operator * (double src) const {
	if (m_pTensor == NULL) THROW_TEMP;
	(*m_pTensor) *= src;
	return *this;
}

CKDTempTensorXXX CKDTempTensorXXX::operator / (double src) const {
	if (m_pTensor == NULL) THROW_TEMP;
	(*m_pTensor) /= src;
	return *this;
}

CKDTempTensorXXX CKDTempTensorXXX::operator * (CKDTempTensorXXX value) const {
	CKDMatrix* matrix1 = m_pTensor->ToMatrix();
	CKDMatrix* matrix2 = value.m_pTensor->ToMatrix(false);
	CKDVector* vector2 = value.m_pTensor->ToVector(false);

	if (matrix2) return CKDTempTensorXXX((*matrix1)*(*matrix2));
	else if (vector2) return CKDTempTensorXXX((*matrix1)*(*vector2));
	else THROW_TEMP;
}

double& CKDTempTensorXXX::operator [](int index) {
	if (m_pTensor == NULL) THROW_TEMP;
	return (*m_pTensor)[index];
}

double CKDTempTensorXXX::operator [](int index) const {
	if (m_pTensor == NULL) THROW_TEMP;
	return (*m_pTensor)[index];
}

CKDTempTensorXXX CKDTempTensorXXX::sigmoid() const {
	if (m_pTensor == NULL) THROW_TEMP;
	return CKDTempTensorXXX(m_pTensor->sigmoid());
}

CKDTempTensorXXX CKDTempTensorXXX::sigmoid_diff() const {
	if (m_pTensor == NULL) THROW_TEMP;
	return CKDTempTensorXXX(m_pTensor->sigmoid_diff());
}

CKDTempTensorXXX CKDTempTensorXXX::relu() const {
	if (m_pTensor == NULL) THROW_TEMP;
	return CKDTempTensorXXX(m_pTensor->relu());
}

CKDTempTensorXXX CKDTempTensorXXX::relu_diff() const {
	if (m_pTensor == NULL) THROW_TEMP;
	return CKDTempTensorXXX(m_pTensor->relu_diff());
}

CKDTempTensorXXX CKDTempTensorXXX::softmax() const {
	if (m_pTensor == NULL) THROW_TEMP;
	return CKDTempTensorXXX(m_pTensor->softmax());
}

CKDTempTensorXXX CKDTempTensorXXX::softmax_diff(const CKDTempTensorXXX vector, const CKDTempTensorXXX& answer) const {
	if (m_pTensor == NULL) THROW_TEMP;
	return CKDTempTensorXXX(m_pTensor->softmax());
}

CKDTempTensorXXX CKDTempTensorXXX::square() const {
	if (m_pTensor == NULL) THROW_TEMP;
	return CKDTempTensorXXX(m_pTensor->softmax());
}

CKDTempTensorXXX CKDTempTensorXXX::fill(double value) const {
	if (m_pTensor == NULL) THROW_TEMP;
	return CKDTempTensorXXX(m_pTensor->softmax());
}

CKDTempTensorXXX CKDTempTensorXXX::mult_each(const CKDTempTensorXXX vector) const {
	if (m_pTensor == NULL) THROW_TEMP;
	return CKDTempTensorXXX(m_pTensor->softmax());
}

CKDTempTensorXXX CKDTempTensorXXX::mult_cross(const CKDTempTensorXXX vector) const {
	if (m_pTensor == NULL) THROW_TEMP;
	return CKDTempTensorXXX(m_pTensor->softmax());
}

CKDTempTensorXXX CKDTempTensorXXX::to_diagonal() const {
	if (m_pTensor == NULL) THROW_TEMP;
	return CKDTempTensorXXX(m_pTensor->softmax());
}
*/