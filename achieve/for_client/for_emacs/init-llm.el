;;(require-package 'request)
;;(require-package 'json)


;; (defun *claude/send-region (beg end)
;;   (make-process
;;    :name "cluade/service-outer"
;;    :buffer "*cluade/service-outer*"
;;    :coding 'utf-8
;;    :command '("python" "/Users/zhangjing/scripts/claude.py")
;;    :filter  (lambda (proc str)
;;               (progn
;;                 (insert-before-markers str))))
;;   (process-send-region  "cluade/service-outer" beg end)
;;   (process-send-string  "cluade/service-outer" "\n\n")
;;   (process-send-eof "cluade/service-outer"))

(defun chatgpt/send-org ()
  (interactive)
  (org-mark-subtree)
  (let  ((begin (region-beginning))
         (end (region-end)))
    (goto-char end)
    (make-process
     :name "cluade/service-outer"
     :buffer "*cluade/service-outer*"
     :coding 'utf-8
     :command '("python" "/home/neromous/Scripts/chatgpt.py")
     :filter  (lambda (proc str)
                (insert str)))
    (process-send-string  "cluade/service-outer" (buffer-substring begin end))
    (process-send-string  "cluade/service-outer" "\n\n")
    (process-send-eof "cluade/service-outer")))



(defun claude/send-org ()
  (interactive)
  (org-mark-subtree)
  (let  ((begin (region-beginning))
         (end (region-end)))
    (goto-char end)
    (make-process
     :name "cluade/service-outer"
     :buffer "*cluade/service-outer*"
     :coding 'utf-8
     :command '("python" "/home/neromous/Scripts/claude.py")
     :filter  (lambda (proc str)
                (insert str)))
    (process-send-string  "cluade/service-outer" (buffer-substring begin end))
    (process-send-string  "cluade/service-outer" "\n\n")
    (process-send-eof "cluade/service-outer")))
;;error in process filter: let: Symbol’s value as variable is void: end

(defun rwkv-7b/send-org ()
  (interactive)
  (org-mark-subtree)
  (let  ((begin (region-beginning))
         (end (region-end)))
    (goto-char end)
    (make-process
     :name "rwkv/service-outer"
     :buffer "*rwkv/service-outer*"
     :coding 'utf-8
     :command '("python" "/home/neromous/Scripts/rwkv_7b.py")
     :filter  (lambda (proc str)
                (insert str)))
    (process-send-string  "rwkv/service-outer" (buffer-substring begin end))
    (process-send-string  "rwkv/service-outer" "\n\n")
    (process-send-eof "rwkv/service-outer")))

(defun rwkv-3b/send-org ()
  (interactive)
  (org-mark-subtree)
  (let  ((begin (region-beginning))
         (end (region-end)))
    (goto-char end)
    (make-process
     :name "rwkv/service-outer-3b1"
     :buffer "*rwkv/service-outer-3b1*"
     :coding 'utf-8
     :command '("python" "/home/neromous/Scripts/rwkv_3b.py")
     :filter  (lambda (proc str)
                (insert str)))
    (process-send-string  "rwkv/service-outer-3b1" (buffer-substring begin end))
    (process-send-string  "rwkv/service-outer-3b1" "\n\n")
    (process-send-eof "rwkv/service-outer-3b1")))


(defun rwkv-7b/train-org ()
  (interactive)
  (org-mark-subtree)
  (let  ((begin (region-beginning))
         (end (region-end)))
    (goto-char end)
    (make-process
     :name "rwkv/service-outer"
     :buffer "*rwkv/service-outer*"
     :coding 'utf-8
     :command '("python" "/home/neromous/Scripts/rwkv_train.py")
     :filter  (lambda (proc str)
                (insert str)))
    (process-send-region  "rwkv/service-outer" begin end)
    (process-send-string  "rwkv/service-outer" "\n\n")
    (process-send-eof "rwkv/service-outer")))

(defun rwkv-3b/train-org ()
  (interactive)
  (org-mark-subtree)
  (let  ((begin (region-beginning))
         (end (region-end)))
    (goto-char end)
    (make-process
     :name "rwkv/service-outer"
     :buffer "*rwkv/service-outer*"
     :coding 'utf-8
     :command '("python" "/home/neromous/Scripts/trian_3b.py")
     :filter  (lambda (proc str)
                (insert str)))
    (process-send-region  "rwkv/service-outer" begin end)
    (process-send-string  "rwkv/service-outer" "\n\n")
    (process-send-eof "rwkv/service-outer")))


;; (require-package 'request)
;; (require-package 'json)

;; ;;
;; (defun llm/reset ()
;;   (request "http://0.0.0.0:8000/reset"
;;     :parser 'json-read
;;     :success (cl-function
;;               (lambda (&key data &allow-other-keys)
;;                 (message "====reset llm")))))

;; (defun llm/load-url (url)
;;   (request url
;;     :parser 'json-read
;;     :success (cl-function
;;               (lambda (&key data &allow-other-keys)
;;                 (message "====reset llm")))))

;; (defun llm/current-text ()
;;   (buffer-substring-no-properties
;;    (point-min)
;;    (point)))


;; (defun rwkv/send-msg-fn (msg)
;;   (make-process
;;    :name "llm/service-outer"
;;    :buffer "*llm/service-outer*"
;;    :coding 'utf-8
;;    :command '("/home/neromous/.miniconda3/envs/blackfog/bin/python"
;;               "/mnt/develop/apps/pyblackfog/api/client.py"))
;;   (process-send-string "llm/service-outer" msg)
;;   (process-send-eof "llm/service-outer"))

;; (defun chatgpt/send-msg-fn (msg)
;;   (make-process
;;    :name "llm/service-outer"
;;    :buffer "*llm/service-outer*"
;;    :coding 'utf-8
;;    :command '("/home/neromous/.miniconda3/envs/blackfog/bin/python"
;;               "/apps/app/pyblackfog/api/chatgpt.py"))
;;   (process-send-string "llm/service-outer" msg)
;;   (process-send-eof "llm/service-outer"))

;; (defun rwkv/send-region (beg end)
;;   (interactive (list (region-beginning)
;;                      (region-end)))

;;   (make-process
;;    :name "llm/service-outer"
;;    :buffer "*llm/service-outer*"
;;    :coding 'utf-8
;;    :command '("/home/neromous/.miniconda3/envs/blackfog/bin/python"
;;               "/home/neromous/Scripts/client.py")
;;    :filter  (lambda (proc str) (insert str)))
;;   (process-send-region "llm/service-outer" beg end)
;;   (process-send-string  "llm/service-outer" "\n\n")
;;   (process-send-eof "llm/service-outer"))


;; (defun rwkv/send-region-background (beg end)
;;   (interactive (list (region-beginning)
;;                      (region-end)))

;;   (make-process
;;    :name "llm/service-outer"
;;    :buffer "*llm/service-outer*"
;;    :coding 'utf-8
;;    :command '("/home/neromous/.miniconda3/envs/blackfog/bin/python"
;;               "/mnt/develop/apps/pyblackfog/api/client.py")
;;    ;;:filter  (lambda (proc str) (insert str))
;;    )
;;   (process-send-region "llm/service-outer" beg end)
;;   (process-send-string  "llm/service-outer" "\n\n")
;;   (process-send-eof "llm/service-outer"))


;; (defun rwkv/send-file (beg end)
;;   (interactive (list (region-beginning)
;;                      (region-end)))
;;   (make-process
;;    :name "llm/service-outer"
;;    :buffer "*llm/service-outer*"
;;    :coding 'utf-8
;;    :command '("/home/neromous/.miniconda3/envs/blackfog/bin/python"
;;               "/mnt/develop/apps/pyblackfog/api/inference_text.py"))
;;   (process-send-region "llm/service-outer" beg end)
;;   (process-send-string  "llm/service-outer" "\n\n")
;;   (process-send-eof "llm/service-outer"))


;; (defun chatgpt/send-region (beg end)
;;   (interactive (list (region-beginning)
;;                      (region-end)))
;;   (make-process
;;    :name "llm/service-outer"
;;    :buffer "*llm/service-outer*"
;;    :coding 'utf-8
;;    :command '("/home/neromous/.miniconda3/envs/blackfog/bin/python"
;;               "/home/neromous/Scripts/chatgpt.py")
;;    :filter  (lambda (proc str) (insert str)))
;;   (process-send-region  "llm/service-outer" beg end)
;;   (process-send-string  "llm/service-outer" "\n\n")
;;   (process-send-eof "llm/service-outer"))

;; (defun rwkv/send-buffer ()
;;   (interactive )
;;   (make-process
;;    :name "llm/service-outer"
;;    :buffer "*llm/service-outer*"
;;    :coding 'utf-8
;;    :command '("/home/neromous/.miniconda3/envs/blackfog/bin/python"
;;               "/home/neromous/Scripts/inference_text.py")
;;    :filter  (lambda (proc str) (insert str)))
;;   (process-send-region "llm/service-outer" (point-min) (point))
;;   (process-send-eof "llm/service-outer"))

;; (defun chatgpt/send-buffer ()
;;   (interactive)
;;   (make-process
;;    :name "llm/service-outer"
;;    :buffer "*llm/service-outer*"
;;    :coding 'utf-8
;;    :command '("/home/neromous/.miniconda3/envs/rocm/bin/python"
;;               "/home/neromous/Scripts/chatgpt.py")
;;    :filter  (lambda (proc str) (insert str)))
;;   (process-send-region  "llm/service-outer" (point-min) (point-max))
;;   (process-send-eof "llm/service-outer"))

(provide 'init-llm)


;; ;; (defun llm/prompt-7b (text)
;; ;;   (request "http://0.0.0.0:8000/generate"
;; ;;     :type "POST"
;; ;;     :data (json-encode  (cons (cons "prompt" text)
;; ;;                               '(("temperature" . 1.2)
;;                                 ("top_p" .  0.7)
;;                                 ("str_stop" .  "\n\n")
;;                                 ("token_count" .  128)
;;                                 ("chunk_len" .  128)

;;                                 )
;;                               ) )
;;     :headers '(("Content-Type" . "application/json"))
;;     :parser 'json-read
;;     :encoding 'utf-8
;;     :success (cl-function
;;               (lambda (&key data &allow-other-keys)
;;                 (insert (assoc-default 'response data))))))



;; (defun llm/prompt-3b (text)
;;   (request "http://0.0.0.0:8000/3b"
;;     :type "POST"
;;     :data (json-encode  (cons (cons "prompt" text)
;;                               '(("temperature" . 1.2)
;;                                 ("top_p" .  0.7)
;;                                 ("str_stop" .  "\n\n")
;;                                 ("token_count" .  128)
;;                                 ("chunk_len" .  128))))
;;     :headers '(("Content-Type" . "application/json"))
;;     :parser 'json-read
;;     :encoding 'utf-8
;;     :success (cl-function
;;               (lambda (&key data &allow-other-keys)
;;                 (insert (assoc-default 'response data))))))
