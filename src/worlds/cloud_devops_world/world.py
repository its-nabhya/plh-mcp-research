from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
from src.worlds.common_yaml_world import YamlStateWorld

class CloudDevOpsWorld(YamlStateWorld):
    tools_file = Path(__file__).parent / 'tools.yaml'
    initial_state = {'services': {'api': {'deps_ok': True, 'deployments': {'staging': [], 'prod': []}, 'replicas': {'staging':1,'prod':2}}}}
    def _pre_deploy_service(self, p, v):
        s=self.state['services'].get(p['service'])
        if not s: self._add_state_violation(v,'Unknown service','service'); return False
        if p['environment'] not in ('staging','prod'): self._add_state_violation(v,'Invalid environment','environment'); return False
        if not s['deps_ok']: self._add_state_violation(v,'Dependencies unhealthy'); return False
        return True
    def _pre_get_logs(self, p, v): return self._pre_deploy_service({'service':p['service'],'environment':p['environment']},v)
    def _pre_scale_service(self, p, v):
        if p['replicas'] < 1: self._add_state_violation(v,'replicas must be >=1','replicas'); return False
        return self._pre_deploy_service(p,v)
    def _pre_rollback_deployment(self, p, v):
        if not self._pre_deploy_service(p,v): return False
        hist=self.state['services'][p['service']]['deployments'][p['environment']]
        if p['deployment_id'] not in [d['id'] for d in hist]: self._add_state_violation(v,'deployment_id not found','deployment_id'); return False
        return True
    def _execute(self, t:str, p:Dict[str, Any], role:str):
        svc=self.state['services'][p['service']]
        env=p['environment']
        if t=='deploy_service':
            did=f"dep-{env}-{len(svc['deployments'][env])+1:03d}"; svc['deployments'][env].append({'id':did,'version':p['version']}); return self._ok({'deployment_id':did})
        if t=='get_logs': return self._ok([f"{p['service']}:{env}:INFO:line{i}" for i in range(1,(p.get('lines',5))+1)])
        if t=='scale_service': svc['replicas'][env]=p['replicas']; return self._ok({'replicas':p['replicas']})
        if t=='rollback_deployment': return self._ok({'rolled_back_to':p['deployment_id']})
        return self._err('NotImplementedError', f'Unknown tool: {t}')

def create_cloud_devops_world(role='developer'):
    return CloudDevOpsWorld({'name':'cloud_devops_world','version':'1.0','complexity':'medium-large','default_user_role':role})
