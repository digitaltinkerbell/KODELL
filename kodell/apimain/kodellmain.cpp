#include "stdafx.h"

#define _EXPORT_KONAN_DEEP_LEARNING_LIB_

#include "../api/kodellapi.h"
#include "../config/config_manager.h"
#include "../project/project_manager.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

namespace Kodell
{
	_KODELL_ ConfigManager::ConfigManager() {
		handle = (KHANDLE)(new CKDConfigManager());
	}

	_KODELL_ ConfigManager::~ConfigManager() {
		delete (CKDConfigManager*) handle;
	}

	void _KODELL_ ConfigManager::Open(const char* filepath) {
		((CKDConfigManager*)handle)->Open(filepath);
	}

	void _KODELL_ ConfigManager::Close() {
		((CKDConfigManager*)handle)->Close();
	}

	KHANDLE _KODELL_ ConfigManager::GetRoot() {
		return ((CKDConfigManager*)handle)->GetRoot();
	}

	int _KODELL_ ConfigManager::GetChildCount(KHANDLE node) {
		return ((CKDConfigManager*)handle)->GetChildCount((XMLNode*)node);
	}

	KHANDLE _KODELL_ ConfigManager::GetParent(KHANDLE node) {
		return ((CKDConfigManager*)handle)->GetParent((XMLNode*)node);
	}

	KHANDLE _KODELL_ ConfigManager::GetNthChild(KHANDLE node, int nth) {
		return ((CKDConfigManager*)handle)->GetNthChild((XMLNode*)node, nth);
	}

	void _KODELL_ ConfigManager::GetNodeInfo(KHANDLE node, ConfigNodeInfo* pInfo) {
		((CKDConfigManager*)handle)->GetNodeInfo((XMLNode*)node, pInfo);
	}
	
	_KODELL_ Project::Project(ConfigManager* pCondMan, KHANDLE node) {
		const char* name = ((XMLElement*)node)->Attribute("name");
		handle = (KHANDLE)(new CKDProjectManager((CKDConfigManager*)pCondMan->handle, name));
	}

	_KODELL_ Project::~Project() {
		delete (CKDProjectManager*)handle;
	}

	void _KODELL_ Project::Execute() {
		((CKDProjectManager*)handle)->Execute();
	}
}
