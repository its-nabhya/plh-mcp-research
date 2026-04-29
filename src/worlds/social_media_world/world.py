from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
from src.worlds.common_yaml_world import YamlStateWorld

class SocialMediaWorld(YamlStateWorld):
    tools_file = Path(__file__).parent / 'tools.yaml'
    initial_state = {'users': {'u1': {'dms_open': True}, 'u2': {'dms_open': False}}, 'posts': [], 'follows': set()}
    def _users_ok(self, ids, v):
        ok=True
        for f in ids:
            if f not in self.state['users']: self._add_state_violation(v,f'Unknown user {f}'); ok=False
        return ok
    def _pre_post_message(self,p,v): return self._users_ok([p['user_id']],v)
    def _pre_get_feed(self,p,v): return self._users_ok([p['user_id']],v)
    def _pre_follow_user(self,p,v): return self._users_ok([p['follower_id'],p['target_id']],v)
    def _pre_send_message(self,p,v):
        if not self._users_ok([p['sender_id'],p['recipient_id']],v): return False
        if not self.state['users'][p['recipient_id']]['dms_open']: self._add_state_violation(v,'Recipient privacy restricts DM','recipient_id'); return False
        return True
    def _execute(self,t:str,p:Dict[str, Any], role:str):
        if t=='post_message': self.state['posts'].append({'user_id':p['user_id'],'text':p['text']}); return self._ok({'post_id':len(self.state['posts'])})
        if t=='get_feed': return self._ok(self.state['posts'][-p.get('limit',10):])
        if t=='follow_user': self.state['follows'].add((p['follower_id'],p['target_id'])); return self._ok({'following':True})
        if t=='send_message': return self._ok({'delivered':True})
        return self._err('NotImplementedError', f'Unknown tool: {t}')

def create_social_media_world(role='user'):
    return SocialMediaWorld({'name':'social_media_world','version':'1.0','complexity':'small','default_user_role':role})
