#include "stdafx.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../math/dimension.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

CKDDimension::CKDDimension() {
	m_pCore = NULL;
}

CKDDimension::CKDDimension(int dim, int size) {
	m_pCore = new CKDDimensionCore;

	m_pCore->m_nDimension = dim;
	m_pCore->m_nSize = 1;

	if (dim) {
		m_pCore->m_pnSize = new int[dim];
		for (int i = 0; i < dim; i++) {
			m_pCore->m_pnSize[i] = size;
			m_pCore->m_nSize *= size;
		}
	}
}

CKDDimension::CKDDimension(int dim, int size1, int size2) {
	if (dim < 2) THROW_TEMP;

	m_pCore = new CKDDimensionCore;

	m_pCore->m_nDimension = dim;
	m_pCore->m_nSize = 1;
	m_pCore->m_pnSize = new int[dim];
	
	m_pCore->m_pnSize[0] = size1;
	m_pCore->m_nSize *= size1;

	for (int i = 1; i < dim; i++) {
		m_pCore->m_pnSize[i] = size2;
		m_pCore->m_nSize *= size2;
	}
}

CKDDimension::CKDDimension(int dim, int size1, int size2, int size3) {
	if (dim < 3) THROW_TEMP;

	m_pCore = new CKDDimensionCore;

	m_pCore->m_nDimension = dim;
	m_pCore->m_nSize = 1;
	m_pCore->m_pnSize = new int[dim];

	m_pCore->m_pnSize[0] = size1;
	m_pCore->m_nSize *= size1;

	m_pCore->m_pnSize[1] = size2;
	m_pCore->m_nSize *= size2;

	for (int i = 2; i < dim; i++) {
		m_pCore->m_pnSize[i] = size3;
		m_pCore->m_nSize *= size3;
	}
}

CKDDimension::CKDDimension(const CKDDimension &src) {
	m_pCore = src.m_pCore;
	if (m_pCore) m_pCore->m_nRefCount++;
}

CKDDimension::~CKDDimension() {
	reset();
}

CKDDimension CKDDimension::clone() const {
	CKDDimension dimension;
	
	dimension.m_pCore = new CKDDimensionCore;
	dimension.m_pCore->m_nDimension = m_pCore->m_nDimension;
	dimension.m_pCore->m_nSize = m_pCore->m_nSize;

	if (m_pCore->m_nDimension) {
		dimension.m_pCore->m_pnSize = new int[m_pCore->m_nDimension];
		memcpy(dimension.m_pCore->m_pnSize, m_pCore->m_pnSize, sizeof(int)*m_pCore->m_nDimension);
	}

	return dimension;
}

CKDDimension CKDDimension::appendAxis(int size) const {
	CKDDimension dimension;
	
	dimension.m_pCore = new CKDDimensionCore;
	dimension.m_pCore->m_nDimension = m_pCore->m_nDimension+1;
	dimension.m_pCore->m_nSize = m_pCore->m_nSize * size;

	dimension.m_pCore->m_pnSize = new int[m_pCore->m_nDimension + 1];

	if (m_pCore->m_nDimension) {
		memcpy(dimension.m_pCore->m_pnSize, m_pCore->m_pnSize, sizeof(int)*m_pCore->m_nDimension);
	}

	dimension.m_pCore->m_pnSize[m_pCore->m_nDimension] = size;

	return dimension;
}

CKDDimension CKDDimension::popTail() const {
	CKDDimension dimension;

	int size = m_pCore->m_pnSize[m_pCore->m_nDimension - 1];

	dimension.m_pCore = new CKDDimensionCore;
	dimension.m_pCore->m_nDimension = m_pCore->m_nDimension - 1;
	dimension.m_pCore->m_nSize = m_pCore->m_nSize / size;

	dimension.m_pCore->m_pnSize = new int[m_pCore->m_nDimension - 1];

	if (m_pCore->m_nDimension > 1) {
		memcpy(dimension.m_pCore->m_pnSize, m_pCore->m_pnSize, sizeof(int)*(m_pCore->m_nDimension - 1));
	}

	return dimension;
}

CKDDimension CKDDimension::popHead() const {
	CKDDimension dimension;

	int size = m_pCore->m_pnSize[0];

	dimension.m_pCore = new CKDDimensionCore;
	dimension.m_pCore->m_nDimension = m_pCore->m_nDimension - 1;
	dimension.m_pCore->m_nSize = m_pCore->m_nSize / size;

	dimension.m_pCore->m_pnSize = new int[m_pCore->m_nDimension - 1];

	memcpy(dimension.m_pCore->m_pnSize, m_pCore->m_pnSize+1, sizeof(int)*(m_pCore->m_nDimension - 1));

	return dimension;
}

const CKDDimension& CKDDimension::operator = (const CKDDimension& src) {
	if (this == &src) return src;

	reset();

	m_pCore = src.m_pCore;
	if (m_pCore) m_pCore->m_nRefCount++;

	return *this;
}

bool CKDDimension::operator == (const CKDDimension& src) const {
	if (this == &src) return true;
	if (m_pCore == src.m_pCore) return true;
	
	if (m_pCore->m_nDimension != src.m_pCore->m_nDimension) return false;
	if (m_pCore->m_nSize != src.m_pCore->m_nSize) return false;

	for (int i = 0; i < m_pCore->m_nDimension; i++) {
		if (m_pCore->m_pnSize[i] != src.m_pCore->m_pnSize[i]) return false;
	}

	return true;
}

bool CKDDimension::operator != (const CKDDimension& src) const {
	return !(*this == src);
}

bool CKDDimension::isEmpty() const {
	return m_pCore == NULL;
}

bool CKDDimension::include_tail(const CKDDimension& src) const {
	int diff = m_pCore->m_nDimension - src.m_pCore->m_nDimension;
	if (diff < 0) return false;
	return memcmp(m_pCore->m_pnSize + diff, src.m_pCore->m_pnSize, src.m_pCore->m_nDimension) == 0;
}

void CKDDimension::reset(int dim) {
	if (m_pCore && --m_pCore->m_nRefCount <= 0) {
		delete m_pCore;
		m_pCore = NULL;
	}

	if (dim <= 0) return;

	m_pCore = new CKDDimensionCore;
	m_pCore->m_nDimension = dim;
	m_pCore->m_pnSize = new int[dim];
	memset(m_pCore->m_pnSize, 0, sizeof(int)*dim);
	m_pCore->m_nSize = 0;
}

void CKDDimension::setSize(int size) {
	reset(1);

	m_pCore->m_pnSize[0] = size;
	m_pCore->m_nSize = size;
}

void CKDDimension::setSize(int size1, int size2) {
	reset(2);

	m_pCore->m_pnSize[0] = size1;
	m_pCore->m_pnSize[1] = size2;
	m_pCore->m_nSize = size1 * size2;
}

void CKDDimension::setSize(int size1, int size2, int size3) {
	reset(3);

	m_pCore->m_pnSize[0] = size1;
	m_pCore->m_pnSize[1] = size2;
	m_pCore->m_pnSize[2] = size3;
	m_pCore->m_nSize = size1 * size2 * size3;
}

void CKDDimension::setAxisSize(int nAxis, int size) {
	if (nAxis < 0 || nAxis >= m_pCore->m_nDimension) THROW_TEMP;

	m_pCore->m_nSize /= m_pCore->m_pnSize[nAxis];
	m_pCore->m_pnSize[nAxis] = size;
	m_pCore->m_nSize *= m_pCore->m_pnSize[nAxis];
}

int CKDDimension::getDimension() const {
	if (!m_pCore) THROW_TEMP;
	return m_pCore->m_nDimension;
}

int CKDDimension::getAxisSize(int nAxis) const {
	if (!m_pCore) THROW_TEMP;
	if (nAxis < 0 || nAxis >= m_pCore->m_nDimension) THROW_TEMP;
	return m_pCore->m_pnSize[nAxis];
}
