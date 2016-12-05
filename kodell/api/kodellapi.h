#pragma once

#undef _KONAN_DEEP_LEARNING_LIB_

#ifdef _LINUX_
	#define _KODELL_
#else
#ifdef _EXPORT_KONAN_DEEP_LEARNING_LIB_
	#define _KODELL_ __declspec(dllexport)
#else	
	#define _KODELL_ 
#endif
#endif

#include "kodelldatatypes.h"

namespace Kodell
{
	class ConfigManager
	{
	public:
		_KODELL_ ConfigManager();
		_KODELL_ ~ConfigManager();

		void _KODELL_ Open(const char* filepath);
		void _KODELL_ Close();

		KHANDLE _KODELL_ GetRoot();
		
		int _KODELL_ GetChildCount(KHANDLE node);
		KHANDLE _KODELL_ GetParent(KHANDLE node);
		KHANDLE _KODELL_ GetNthChild(KHANDLE node, int nth);
		void _KODELL_ GetNodeInfo(KHANDLE node, ConfigNodeInfo* pInfo);

		KHANDLE handle;
	};

	class Project
	{
	public:
		_KODELL_ Project(ConfigManager* pCondMan, KHANDLE node);
		_KODELL_ ~Project();

		void _KODELL_ Execute();

		KHANDLE handle;
	};
}
