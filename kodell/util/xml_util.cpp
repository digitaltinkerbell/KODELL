#include "stdafx.h"

#include "../util/xml_util.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

XMLNode* CKDXmlUtil::SeekTagChild(XMLNode* parent, const char* tag) {
	if (parent == NULL) return NULL;
	XMLNode *child;
	for (child = parent->FirstChildElement(); child; child = child->NextSiblingElement()) {
		if (strcmp(child->Value(), tag) == 0) return child;
	}
	return NULL;
}

XMLNode* CKDXmlUtil::SeekTagSibling(XMLNode* sibling, const char* tag) {
	if (sibling == NULL) return NULL;
	XMLNode *node;
	for (node = sibling->NextSiblingElement(); node; node = node->NextSiblingElement()) {
		if (strcmp(node->Value(), tag) == 0) return node;
	}
	return NULL;
}

XMLNode* CKDXmlUtil::SeekTagNamedChild(XMLNode* parent, const char* tag, const char* name) {
	if (parent == NULL) return NULL;
	XMLNode *child;
	for (child = parent->FirstChildElement(); child; child = child->NextSiblingElement()) {
		if (strcmp(child->Value(), tag) != 0) continue;
		if (strcmp(child->ToElement()->Attribute("name"), name) != 0) continue;

		return child;
	}
	return NULL;
}

const char* CKDXmlUtil::GetAttribute(XMLNode* node, const char* attr) {
	if (node == NULL) return NULL;
	return node->ToElement()->Attribute(attr);
}

int CKDXmlUtil::GetIntAttr(XMLNode* node, const char* attr, int defVal) {
	if (node == NULL) return defVal;
	const char* value = node->ToElement()->Attribute(attr);
	return value ? atoi(value) : defVal;
}

double CKDXmlUtil::GetDoubleAttr(XMLNode* node, const char* attr) {
	if (node == NULL) return 0;
	const char* value = node->ToElement()->Attribute(attr);
	return value ? atof(value) : 0;
}

int CKDXmlUtil::GetEnumTag(XMLNode* node, const char* pNames[]) {
	if (node == NULL) return -1;
	const char* name = node->Value();
	if (name == NULL) return -1;
	for (int i = 0; pNames[i]; i++) {
		if (strcmp(pNames[i], name) == 0) return i;
	}

	return -1;	// enum_xxx_undef
}

int CKDXmlUtil::GetEnumAttr(XMLNode* node, const char* attr, const char* pNames[]) {
	if (node == NULL) return -1;
	const char* name = node->ToElement()->Attribute(attr);
	if (name == NULL) return -1;
	for (int i = 0; pNames[i]; i++) {
		if (strcmp(pNames[i], name) == 0) return i;
	}

	return -1;	// enum_xxx_undef
}

int CKDXmlUtil::GetChildCount(XMLNode* node) {
	if (node == NULL) return 0;
	XMLNode *child;
	int count = 0;
	for (child = node->FirstChildElement(); child; child = child->NextSiblingElement()) count++;
	return count;
}

XMLNode* CKDXmlUtil::GetNthChild(XMLNode* node, int nth) {
	if (node == NULL) return 0;
	XMLNode *child;
	for (child = node->FirstChildElement(); child && nth > 0; child = child->NextSiblingElement()) nth--;
	return child;
}

int CKDXmlUtil::GetTagChildCount(XMLNode* node, const char* tag) {
	if (node == NULL) return 0;
	XMLNode *child;
	int count = 0;
	for (child = node->FirstChildElement(); child; child = child->NextSiblingElement()) {
		if (strcmp(child->Value(), tag) == 0) count++;
	}
	return count;
}

XMLNode* CKDXmlUtil::SeekDefClause(XMLNode* node, string name) {
	const char* pname = name.c_str();

	XMLNode* dataset_node = CKDXmlUtil::SeekTagChild(node, pname);
	if (dataset_node != NULL) return dataset_node;

	string refname = name + "ref";
	string setname = name + "s";
	
	const char* pref = refname.c_str();
	const char* pset = setname.c_str();

	XMLNode* ref_node = CKDXmlUtil::SeekTagChild(node, pref);
	if (ref_node != NULL) {
		const char* nameAttr = ref_node->ToElement()->Attribute("name");

		const char* p1 = node->Parent()->Value();
		const char* p2 = node->Parent()->Parent()->Value();

		XMLNode* datasets = CKDXmlUtil::SeekTagChild(node->Parent()->Parent(), pset);
		XMLNode* dataset = CKDXmlUtil::SeekTagNamedChild(datasets, pname, nameAttr);

		if (dataset) return dataset;
	}

	return NULL;
}

