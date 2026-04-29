from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
from src.worlds.common_yaml_world import YamlStateWorld

class HealthcareWorld(YamlStateWorld):
    tools_file = Path(__file__).parent / 'tools.yaml'
    initial_state = {
        'patients': {
            'P100': {'name': 'Alice Kim', 'medications': ['atorvastatin'], 'labs': {'cbc': 'normal'}},
            'P200': {'name': 'Bob Lee', 'medications': [], 'labs': {'a1c': '6.5'}},
        },
        'appointments': {}
    }
    def _pre_get_patient_record(self, p, v):
        ok = p['patient_id'] in self.state['patients']
        if not ok: self._add_state_violation(v, 'Unknown patient_id', 'patient_id')
        return ok
    _pre_prescribe_medication = _pre_get_patient_record
    _pre_get_lab_results = _pre_get_patient_record
    def _pre_schedule_appointment(self, p, v):
        if p['patient_id'] not in self.state['patients']:
            self._add_state_violation(v, 'Unknown patient_id', 'patient_id'); return False
        if p['datetime'] in self.state['appointments']:
            self._add_state_violation(v, 'Appointment slot occupied', 'datetime'); return False
        return True
    def _execute(self, tool_name:str, params:Dict[str, Any], role:str):
        if tool_name=='get_patient_record': return self._ok(self.state['patients'][params['patient_id']])
        if tool_name=='prescribe_medication':
            self.state['patients'][params['patient_id']]['medications'].append(f"{params['medication']}:{params['dose_mg']}mg")
            return self._ok({'patient_id': params['patient_id'], 'prescribed': True})
        if tool_name=='schedule_appointment':
            self.state['appointments'][params['datetime']] = {'patient_id': params['patient_id'], 'clinician_id': params['clinician_id']}
            return self._ok({'scheduled': True, 'slot': params['datetime']})
        if tool_name=='get_lab_results':
            labs = self.state['patients'][params['patient_id']]['labs']
            return self._ok(labs if 'panel' not in params else {params['panel']: labs.get(params['panel'], 'not_found')})
        return self._err('NotImplementedError', f'Unknown tool: {tool_name}')

def create_healthcare_world(role='nurse'):
    return HealthcareWorld({'name':'healthcare_world','version':'1.0','complexity':'medium','default_user_role':role})
