#pragma once

#include <stdio.h>
#include <string.h>

class CKDDimensionCore {
	friend class CKDDimension;

protected:
	CKDDimensionCore() { m_nRefCount = 1;  m_nDimension = 0;  m_pnSize = NULL; }
	~CKDDimensionCore() { delete[] m_pnSize; }
	int m_nRefCount;
	int m_nDimension;
	int m_nSize;
	int *m_pnSize;
};

class CKDDimension {
public:
	CKDDimension();
	CKDDimension(int dim, int size=1);
	CKDDimension(int dim, int size1, int size2);
	CKDDimension(int dim, int size1, int size2, int size3);
	CKDDimension(const CKDDimension &src);

	virtual ~CKDDimension();

	const CKDDimension& operator = (const CKDDimension& src);

	CKDDimension clone() const;

	bool operator == (const CKDDimension& src) const;
	bool operator != (const CKDDimension& src) const;

	bool isEmpty() const;
	bool include_tail(const CKDDimension& src) const;

	int getDimension() const;
	int getSize() const { return m_pCore ? m_pCore->m_nSize : 0; }
	int getAxisSize(int nAxis) const;

	void setSize(int size);
	void setSize(int size1, int size2);
	void setSize(int size1, int size2, int size3);
	void setAxisSize(int nAxis, int size);

	CKDDimension appendAxis(int size) const;
	CKDDimension popTail() const;
	CKDDimension popHead() const;

	void reset(int dim=0);

protected:
	CKDDimensionCore* m_pCore;
};

