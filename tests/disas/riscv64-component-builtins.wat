;;! target = "riscv64"
;;! test = 'optimize'
;;! filter = 'component-resource-drop[0]_wasm_call'

(component
  (type $a (resource (rep i32)))
  (core func $f (canon resource.drop $a))

  (core module $m (import "" "" (func (param i32))))
  (core instance (instantiate $m (with "" (instance (export "" (func $f))))))
)

;; function u0:0(i64 vmctx, i64, i32) tail {
;;     sig0 = (i64 sext, i32 sext, i32 sext) -> i64 sext system_v
;;     sig1 = (i64 sext vmctx) system_v
;;
;; block0(v0: i64, v1: i64, v2: i32):
;;     v3 = load.i32 notrap aligned little v0
;;     v20 = iconst.i32 0x706d_6f63
;;     v4 = icmp eq v3, v20  ; v20 = 0x706d_6f63
;;     trapz v4, user1
;;     v5 = load.i64 notrap aligned v0+16
;;     v6 = get_frame_pointer.i64 
;;     v7 = load.i64 notrap aligned v6
;;     store notrap aligned v7, v5+40
;;     v8 = get_return_address.i64 
;;     store notrap aligned v8, v5+48
;;     v9 = load.i32 notrap aligned readonly v0+32
;;     v19 = iconst.i32 1
;;     v10 = band v9, v19  ; v19 = 1
;;     trapz v10, user26
;;     v12 = load.i64 notrap aligned readonly v0+8
;;     v13 = load.i64 notrap aligned readonly v12+16
;;     v11 = iconst.i32 0
;;     v14 = call_indirect sig0, v13(v0, v11, v2)  ; v11 = 0
;;     v15 = iconst.i64 -1
;;     v16 = icmp ne v14, v15  ; v15 = -1
;;     brif v16, block2, block1
;;
;; block1 cold:
;;     v17 = load.i64 notrap aligned readonly v1+16
;;     v18 = load.i64 notrap aligned readonly v17+416
;;     call_indirect sig1, v18(v1)
;;     trap user1
;;
;; block2:
;;     brif.i64 v14, block3, block4
;;
;; block3:
;;     jump block4
;;
;; block4:
;;     return
;; }
