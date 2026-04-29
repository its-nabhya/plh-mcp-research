from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
from src.worlds.common_yaml_world import YamlStateWorld

class EcommerceWorld(YamlStateWorld):
    tools_file = Path(__file__).parent / 'tools.yaml'
    initial_state = {'catalog': {'SKU1': {'name':'Keyboard','price':50}, 'SKU2': {'name':'Mouse','price':25}}, 'cart': {}, 'orders': {}}
    def _pre_add_to_cart(self, p, v):
        ok = p['product_id'] in self.state['catalog'] and p['quantity'] > 0
        if p['product_id'] not in self.state['catalog']: self._add_state_violation(v,'Unknown product_id','product_id')
        if p['quantity'] <= 0: self._add_state_violation(v,'quantity must be > 0','quantity')
        return ok
    def _pre_checkout(self, p, v):
        ok = len(self.state['cart']) > 0
        if not ok: self._add_state_violation(v,'cart is empty')
        return ok
    def _pre_track_order(self, p, v):
        ok = p['order_id'] in self.state['orders']
        if not ok: self._add_state_violation(v,'order_id not found','order_id')
        return ok
    def _execute(self, t:str, p:Dict[str, Any], role:str):
        if t=='search_product': return self._ok([{'product_id':k, **v} for k,v in self.state['catalog'].items() if p['query'].lower() in v['name'].lower()])
        if t=='add_to_cart': self.state['cart'][p['product_id']] = self.state['cart'].get(p['product_id'],0)+p['quantity']; return self._ok({'cart':self.state['cart']})
        if t=='checkout': oid=f"ORD-{len(self.state['orders'])+1:04d}"; self.state['orders'][oid]={'items':dict(self.state['cart']),'status':'processing'}; self.state['cart']={}; return self._ok({'order_id':oid})
        if t=='track_order': return self._ok(self.state['orders'][p['order_id']])
        return self._err('NotImplementedError', f'Unknown tool: {t}')

def create_ecommerce_world(role='user'):
    return EcommerceWorld({'name':'ecommerce_world','version':'1.0','complexity':'small-medium','default_user_role':role})
