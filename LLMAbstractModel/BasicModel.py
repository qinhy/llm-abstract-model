# from https://github.com/qinhy/singleton-key-value-storage.git
from datetime import datetime
import json
from typing import Optional, Type
import unittest
from uuid import uuid4
from datetime import datetime, timezone
from pydantic import BaseModel, ConfigDict, Field
from .Storage import SingletonKeyValueStorage

def now_utc():
    return datetime.now(timezone.utc)

class BasicModel(BaseModel):
    # def __call__(self, *args, **kwargs):
    #     raise NotImplementedError('This method should be implemented by subclasses.')
    
    def _log_error(self, e):
        return f"[{self.__class__.__name__}] Error: {str(e)}"
    
    def _try_error(self, func, default_value=('NULL',None)):
        try:
            return (True,func())
        except Exception as e:
            self._log_error(e)
            return (False,default_value)
        
    def _try_binary_error(self, func):
        return self._try_error(func)[0]
    
    def _try_obj_error(self, func, default_value=('NULL',None)):
        return self._try_error(func,default_value)[1]
    
class Controller4Basic:
    class AbstractObjController:
        def __init__(self, store, model):
            self.model:Model4Basic.AbstractObj = model
            self._store:BasicStore = store
        
        def storage(self):return self._store

        def update(self, **kwargs):
            assert self.model is not None, 'controller has null model!'
            for key, value in kwargs.items():
                if hasattr(self.model, key):
                    setattr(self.model, key, value)
            self._update_timestamp()
            self.store()
            return self

        def _update_timestamp(self):
            assert self.model is not None, 'controller has null model!'
            self.model.update_time = now_utc()
            
        def store(self):
            assert self.model._id is not None
            self.storage().set(self.model._id,self.model.model_dump_json_dict())
            return self

        def delete(self):
            self.storage().delete(self.model.get_id())
            self.model.controller = None

        def update_metadata(self, key, value):
            updated_metadata = {**self.model.metadata, key: value}
            self.update(metadata = updated_metadata)
            return self
        
    class AbstractGroupController(AbstractObjController):
        def __init__(self, store, model):
            self.model: Model4Basic.AbstractGroup = model
            self._store: BasicStore = store

        def yield_children_recursive(self, depth: int = 0):
            for child_id in self.model.children_id:
                if not self.storage().exists(child_id):
                    continue
                child: Model4Basic.AbstractObj = self.storage().find(child_id)
                if hasattr(child, 'parent_id') and hasattr(child, 'children_id'):
                    group:Controller4Basic.AbstractGroupController = child.controller
                    yield from group.yield_children_recursive(depth + 1)
                yield child, depth

        def delete_recursive(self):
            for child, _ in self.yield_children_recursive():
                child.controller.delete()
            self.delete()

        def get_children_recursive(self):
            children_list = []
            for child_id in self.model.children_id:
                if not self.storage().exists(child_id):
                    continue
                child: Model4Basic.AbstractObj = self.storage().find(child_id)
                if hasattr(child, 'parent_id') and hasattr(child, 'children_id'):
                    group:Controller4Basic.AbstractGroupController = child.controller
                    children_list.append(group.get_children_recursive())
                else:
                    children_list.append(child)            
            return children_list

        def get_children(self):
            assert self.model is not None, 'Controller has a null model!'
            return [self.storage().find(child_id) for child_id in self.model.children_id]

        def get_child(self, child_id: str):
            return self.storage().find(child_id)
        
        def add_child(self, child_id: str):
            return self.update(children_id= self.model.children_id + [child_id])

        def delete_child(self, child_id:str):
            if child_id not in self.model.children_id:return self
            remaining_ids = [cid for cid in self.model.children_id if cid != child_id]
            child_con = self.storage().find(child_id).controller
            if hasattr(child_con, 'delete_recursive'):
                child_con:Controller4Basic.AbstractGroupController = child_con
                child_con.delete_recursive()
            else:
                child_con.delete()
            self.update(children_id = remaining_ids)
            return self

class Model4Basic:
    class AbstractObj(BasicModel):
        _id: str=None
        rank: list = [0]
        create_time: datetime = Field(default_factory=now_utc)
        update_time: datetime = Field(default_factory=now_utc)
        status: str = ""
        metadata: dict = {}
        auto_del: bool = False # auto delete when removed from memory 
          
        # auto exclude when model dump
        model_config = ConfigDict(arbitrary_types_allowed=True)
        controller: Optional[Controller4Basic.AbstractObjController] = None

        def __obj_del__(self):
            # print(f'BasicApp.store().delete({self.id})')
            self.controller.delete()
        
        def __del__(self):
            if hasattr(self,'auto_del') and self.auto_del: self.__obj_del__()
        
        def model_dump_json_dict(self):
            return json.loads(self.model_dump_json())

        def class_name(self): return self.__class__.__name__

        def model_copy(self, *, update = None, deep = False):
            res = super().model_copy(update=update, deep=deep)
            res._id = None
            return res

        def set_id(self,id:str):
            assert self._id is None, 'this obj is been setted! can not set again!'
            self._id = id
            return self
        
        def gen_new_id(self): return f"{self.class_name()}:{uuid4()}"

        def get_id(self):
            assert self._id is not None, 'this obj is not setted!'
            return self._id

        def model_dump_json(self, *, indent = None, include = None, exclude = None, context = None, by_alias = False, exclude_unset = False, exclude_defaults = False, exclude_none = False, round_trip = False, warnings = True, serialize_as_any = False):
            if exclude:
                exclude += ['controller']
            else:
                exclude = ['controller']
            return super().model_dump_json(indent=indent, include=include, exclude=exclude, context=context, by_alias=by_alias, exclude_unset=exclude_unset, exclude_defaults=exclude_defaults, exclude_none=exclude_none, round_trip=round_trip, warnings=warnings, serialize_as_any=serialize_as_any)

        def model_dump(self, *, mode = 'python', include = None, exclude = None, context = None, by_alias = False, exclude_unset = False, exclude_defaults = False, exclude_none = False, round_trip = False, warnings = True, serialize_as_any = False):
            if exclude:
                exclude += ['controller']
            else:
                exclude = ['controller']
            return super().model_dump(mode=mode, include=include, exclude=exclude, context=context, by_alias=by_alias, exclude_unset=exclude_unset, exclude_defaults=exclude_defaults, exclude_none=exclude_none, round_trip=round_trip, warnings=warnings, serialize_as_any=serialize_as_any)

        # def _get_controller_class(self,modelclass=Controller4Basic):
        #     class_type = self.__class__.__name__+'Controller'
        #     res = {c.__name__:c for c in [i for k,i in modelclass.__dict__.items() if '_' not in k]}
        #     res = res.get(class_type, None)
        #     if res is None: raise ValueError(f'No such class of {class_type}')
        #     return res
        
        def _get_controller_class(self,modelclass=Controller4Basic):
            class_type = self.__class__.__name__+'Controller'
            res = {c.__name__:c for c in [i for k,i in modelclass.__dict__.items() if '_' not in k]}
            res = res.get(class_type, None)
            if res is None: print(f'[warning]: No such class of {class_type}, use Controller4Basic.AbstractObjController')
            res = Controller4Basic.AbstractObjController
            return res
        
        def init_controller(self,store):
            self.controller = self._get_controller_class()(store,self)

    class AbstractGroup(AbstractObj):
        author_id: str=''
        parent_id: str = ''
        children_id: list[str] = []

        # auto exclude when model dump
        controller: Optional[Controller4Basic.AbstractGroupController] = None

class BasicStore(SingletonKeyValueStorage):
    MODEL_CLASS_GROUP = Model4Basic
    
    def __init__(self, version_controll=False) -> None:
        super().__init__(version_controll)
        self.python_backend()

    def _get_class(self, id: str, modelclass=MODEL_CLASS_GROUP):
        class_type = id.split(':')[0]
        res = [i for k,i in modelclass.__dict__.items() if '_' not in k]
        res = {c.__name__:c for c in res}
        res = res.get(class_type, None)
        if res is None: raise ValueError(f'No such class of {class_type}')
        return res
    
    def _auto_fix_id(self,obj:MODEL_CLASS_GROUP.AbstractObj, id:str="None"):
        class_type = id.split(':')[0]
        obj_class_type = obj.__class__.__name__
        if class_type != obj_class_type:
            id = f'{obj_class_type}:{id}'
        return id
    
    def _get_as_obj(self,id,data_dict)->MODEL_CLASS_GROUP.AbstractObj:
        obj:Model4Basic.AbstractObj = self._get_class(id)(**data_dict)
        obj.set_id(id).init_controller(self)
        return obj
    
    def _add_new_obj(self, obj:MODEL_CLASS_GROUP.AbstractObj, id:str=None):
        id,d = obj.gen_new_id() if id is None else id, obj.model_dump_json_dict()
        id = self._auto_fix_id(obj,id)
        self.set(id,d)
        return self._get_as_obj(id,d)
    
    def add_new_class(self,obj_class_type:Type[MODEL_CLASS_GROUP.AbstractObj]):
        if not hasattr(self.MODEL_CLASS_GROUP,obj_class_type.__name__):
            setattr(self.MODEL_CLASS_GROUP,obj_class_type.__name__,obj_class_type)
    
    def add_new(self, obj_class_type=MODEL_CLASS_GROUP.AbstractObj,id:str=None):#, id:str=None)->MODEL_CLASS_GROUP.AbstractObj:
        obj_name = obj_class_type.__name__
        if not hasattr(self.MODEL_CLASS_GROUP,obj_name):
            setattr(self.MODEL_CLASS_GROUP,obj_name,obj_class_type)
        def add_obj(*args,**kwargs):
            obj = obj_class_type(*args,**kwargs)
            if obj._id is not None: raise ValueError(f'obj._id is "{obj._id}", must be none')
            return self._add_new_obj(obj,id)
        return add_obj
    
    def add_new_obj(self, obj:MODEL_CLASS_GROUP.AbstractObj, id:str=None)->MODEL_CLASS_GROUP.AbstractObj:
        obj_name = obj.__class__.__name__
        if not hasattr(self.MODEL_CLASS_GROUP,obj_name):
            setattr(self.MODEL_CLASS_GROUP,obj_name,obj.__class__)  
        if obj._id is not None: raise ValueError(f'obj._id is {obj._id}, must be none')
        return self._add_new_obj(obj,id)
    
    def add_new_group(self, obj:Model4Basic.AbstractGroup, id:str=None)->Model4Basic.AbstractGroup:        
        if obj._id is not None: raise ValueError(f'obj._id is {obj._id}, must be none')
        return self._add_new_obj(obj,id)
    
    def find(self,id:str) -> MODEL_CLASS_GROUP.AbstractObj:
        raw = self.get(id)
        if raw is None:
            raws = self.find_all(f'*:{id}')
            if len(raws)==1:
                return raws[0]
            return None
        return self._get_as_obj(id,raw)
    
    def find_all(self,id:str=f'AbstractObj:*')->list[MODEL_CLASS_GROUP.AbstractObj]:
        return [self.find(k) for k in self.keys(id)]
