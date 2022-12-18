<<<<<<< Updated upstream
/* crypto/crypto.h */
/* ====================================================================
 * Copyright (c) 1998-2006 The OpenSSL Project.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer. 
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in
 *    the documentation and/or other materials provided with the
 *    distribution.
 *
 * 3. All advertising materials mentioning features or use of this
 *    software must display the following acknowledgment:
 *    "This product includes software developed by the OpenSSL Project
 *    for use in the OpenSSL Toolkit. (http://www.openssl.org/)"
 *
 * 4. The names "OpenSSL Toolkit" and "OpenSSL Project" must not be used to
 *    endorse or promote products derived from this software without
 *    prior written permission. For written permission, please contact
 *    openssl-core@openssl.org.
 *
 * 5. Products derived from this software may not be called "OpenSSL"
 *    nor may "OpenSSL" appear in their names without prior written
 *    permission of the OpenSSL Project.
 *
 * 6. Redistributions of any form whatsoever must retain the following
 *    acknowledgment:
 *    "This product includes software developed by the OpenSSL Project
 *    for use in the OpenSSL Toolkit (http://www.openssl.org/)"
 *
 * THIS SOFTWARE IS PROVIDED BY THE OpenSSL PROJECT ``AS IS'' AND ANY
 * EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE OpenSSL PROJECT OR
 * ITS CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
 * OF THE POSSIBILITY OF SUCH DAMAGE.
 * ====================================================================
 *
 * This product includes cryptographic software written by Eric Young
 * (eay@cryptsoft.com).  This product includes software written by Tim
 * Hudson (tjh@cryptsoft.com).
 *
 */
/* Copyright (C) 1995-1998 Eric Young (eay@cryptsoft.com)
 * All rights reserved.
 *
 * This package is an SSL implementation written
 * by Eric Young (eay@cryptsoft.com).
 * The implementation was written so as to conform with Netscapes SSL.
 * 
 * This library is free for commercial and non-commercial use as long as
 * the following conditions are aheared to.  The following conditions
 * apply to all code found in this distribution, be it the RC4, RSA,
 * lhash, DES, etc., code; not just the SSL code.  The SSL documentation
 * included with this distribution is covered by the same copyright terms
 * except that the holder is Tim Hudson (tjh@cryptsoft.com).
 * 
 * Copyright remains Eric Young's, and as such any Copyright notices in
 * the code are not to be removed.
 * If this package is used in a product, Eric Young should be given attribution
 * as the author of the parts of the library used.
 * This can be in the form of a textual message at program startup or
 * in documentation (online or textual) provided with the package.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *    "This product includes cryptographic software written by
 *     Eric Young (eay@cryptsoft.com)"
 *    The word 'cryptographic' can be left out if the rouines from the library
 *    being used are not cryptographic related :-).
 * 4. If you include any Windows specific code (or a derivative thereof) from 
 *    the apps directory (application code) you must include an acknowledgement:
 *    "This product includes software written by Tim Hudson (tjh@cryptsoft.com)"
 * 
 * THIS SOFTWARE IS PROVIDED BY ERIC YOUNG ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * 
 * The licence and distribution terms for any publically available version or
 * derivative of this code cannot be changed.  i.e. this code cannot simply be
 * copied and put under another distribution licence
 * [including the GNU Public Licence.]
 */
/* ====================================================================
 * Copyright 2002 Sun Microsystems, Inc. ALL RIGHTS RESERVED.
 * ECDH support in OpenSSL originally developed by 
 * SUN MICROSYSTEMS, INC., and contributed to the OpenSSL project.
 */

#ifndef HEADER_CRYPTO_H
#define HEADER_CRYPTO_H

#include <stdlib.h>

#include <openssl/e_os2.h>

#ifndef OPENSSL_NO_FP_API
#include <stdio.h>
#endif

#include <openssl/stack.h>
#include <openssl/safestack.h>
#include <openssl/opensslv.h>
#include <openssl/ossl_typ.h>

#ifdef CHARSET_EBCDIC
#include <openssl/ebcdic.h>
#endif

/* Resolve problems on some operating systems with symbol names that clash
   one way or another */
#include <openssl/symhacks.h>
=======
/*
 * Copyright 1995-2019 The OpenSSL Project Authors. All Rights Reserved.
 * Copyright (c) 2002, Oracle and/or its affiliates. All rights reserved
 *
 * Licensed under the OpenSSL license (the "License").  You may not use
 * this file except in compliance with the License.  You can obtain a copy
 * in the file LICENSE in the source distribution or at
 * https://www.openssl.org/source/license.html
 */

#ifndef HEADER_CRYPTO_H
# define HEADER_CRYPTO_H

# include <stdlib.h>
# include <time.h>

# include <openssl/e_os2.h>

# ifndef OPENSSL_NO_STDIO
#  include <stdio.h>
# endif

# include <openssl/safestack.h>
# include <openssl/opensslv.h>
# include <openssl/ossl_typ.h>
# include <openssl/opensslconf.h>
# include <openssl/cryptoerr.h>

# ifdef CHARSET_EBCDIC
#  include <openssl/ebcdic.h>
# endif

/*
 * Resolve problems on some operating systems with symbol names that clash
 * one way or another
 */
# include <openssl/symhacks.h>

# if OPENSSL_API_COMPAT < 0x10100000L
#  include <openssl/opensslv.h>
# endif
>>>>>>> Stashed changes

#ifdef  __cplusplus
extern "C" {
#endif

<<<<<<< Updated upstream
/* Backward compatibility to SSLeay */
/* This is more to be used to check the correct DLL is being used
 * in the MS world. */
#define SSLEAY_VERSION_NUMBER	OPENSSL_VERSION_NUMBER
#define SSLEAY_VERSION		0
/* #define SSLEAY_OPTIONS	1 no longer supported */
#define SSLEAY_CFLAGS		2
#define SSLEAY_BUILT_ON		3
#define SSLEAY_PLATFORM		4
#define SSLEAY_DIR		5

/* Already declared in ossl_typ.h */
#if 0
typedef struct crypto_ex_data_st CRYPTO_EX_DATA;
/* Called when a new object is created */
typedef int CRYPTO_EX_new(void *parent, void *ptr, CRYPTO_EX_DATA *ad,
					int idx, long argl, void *argp);
/* Called when an object is free()ed */
typedef void CRYPTO_EX_free(void *parent, void *ptr, CRYPTO_EX_DATA *ad,
					int idx, long argl, void *argp);
/* Called when we need to dup an object */
typedef int CRYPTO_EX_dup(CRYPTO_EX_DATA *to, CRYPTO_EX_DATA *from, void *from_d, 
					int idx, long argl, void *argp);
#endif

/* A generic structure to pass assorted data in a expandable way */
typedef struct openssl_item_st
	{
	int code;
	void *value;		/* Not used for flag attributes */
	size_t value_size;	/* Max size of value for output, length for input */
	size_t *value_length;	/* Returned length of value for output */
	} OPENSSL_ITEM;


/* When changing the CRYPTO_LOCK_* list, be sure to maintin the text lock
 * names in cryptlib.c
 */

#define	CRYPTO_LOCK_ERR			1
#define	CRYPTO_LOCK_EX_DATA		2
#define	CRYPTO_LOCK_X509		3
#define	CRYPTO_LOCK_X509_INFO		4
#define	CRYPTO_LOCK_X509_PKEY		5
#define CRYPTO_LOCK_X509_CRL		6
#define CRYPTO_LOCK_X509_REQ		7
#define CRYPTO_LOCK_DSA			8
#define CRYPTO_LOCK_RSA			9
#define CRYPTO_LOCK_EVP_PKEY		10
#define CRYPTO_LOCK_X509_STORE		11
#define CRYPTO_LOCK_SSL_CTX		12
#define CRYPTO_LOCK_SSL_CERT		13
#define CRYPTO_LOCK_SSL_SESSION		14
#define CRYPTO_LOCK_SSL_SESS_CERT	15
#define CRYPTO_LOCK_SSL			16
#define CRYPTO_LOCK_SSL_METHOD		17
#define CRYPTO_LOCK_RAND		18
#define CRYPTO_LOCK_RAND2		19
#define CRYPTO_LOCK_MALLOC		20
#define CRYPTO_LOCK_BIO			21
#define CRYPTO_LOCK_GETHOSTBYNAME	22
#define CRYPTO_LOCK_GETSERVBYNAME	23
#define CRYPTO_LOCK_READDIR		24
#define CRYPTO_LOCK_RSA_BLINDING	25
#define CRYPTO_LOCK_DH			26
#define CRYPTO_LOCK_MALLOC2		27
#define CRYPTO_LOCK_DSO			28
#define CRYPTO_LOCK_DYNLOCK		29
#define CRYPTO_LOCK_ENGINE		30
#define CRYPTO_LOCK_UI			31
#define CRYPTO_LOCK_ECDSA               32
#define CRYPTO_LOCK_EC			33
#define CRYPTO_LOCK_ECDH		34
#define CRYPTO_LOCK_BN  		35
#define CRYPTO_LOCK_EC_PRE_COMP		36
#define CRYPTO_LOCK_STORE		37
#define CRYPTO_LOCK_COMP		38
#define CRYPTO_LOCK_FIPS		39
#define CRYPTO_LOCK_FIPS2		40
#define CRYPTO_NUM_LOCKS		41

#define CRYPTO_LOCK		1
#define CRYPTO_UNLOCK		2
#define CRYPTO_READ		4
#define CRYPTO_WRITE		8

#ifndef OPENSSL_NO_LOCKING
#ifndef CRYPTO_w_lock
#define CRYPTO_w_lock(type)	\
	CRYPTO_lock(CRYPTO_LOCK|CRYPTO_WRITE,type,__FILE__,__LINE__)
#define CRYPTO_w_unlock(type)	\
	CRYPTO_lock(CRYPTO_UNLOCK|CRYPTO_WRITE,type,__FILE__,__LINE__)
#define CRYPTO_r_lock(type)	\
	CRYPTO_lock(CRYPTO_LOCK|CRYPTO_READ,type,__FILE__,__LINE__)
#define CRYPTO_r_unlock(type)	\
	CRYPTO_lock(CRYPTO_UNLOCK|CRYPTO_READ,type,__FILE__,__LINE__)
#define CRYPTO_add(addr,amount,type)	\
	CRYPTO_add_lock(addr,amount,type,__FILE__,__LINE__)
#endif
#else
#define CRYPTO_w_lock(a)
#define CRYPTO_w_unlock(a)
#define CRYPTO_r_lock(a)
#define CRYPTO_r_unlock(a)
#define CRYPTO_add(a,b,c)	((*(a))+=(b))
#endif

/* Some applications as well as some parts of OpenSSL need to allocate
   and deallocate locks in a dynamic fashion.  The following typedef
   makes this possible in a type-safe manner.  */
/* struct CRYPTO_dynlock_value has to be defined by the application. */
typedef struct
	{
	int references;
	struct CRYPTO_dynlock_value *data;
	} CRYPTO_dynlock;


/* The following can be used to detect memory leaks in the SSLeay library.
 * It used, it turns on malloc checking */

#define CRYPTO_MEM_CHECK_OFF	0x0	/* an enume */
#define CRYPTO_MEM_CHECK_ON	0x1	/* a bit */
#define CRYPTO_MEM_CHECK_ENABLE	0x2	/* a bit */
#define CRYPTO_MEM_CHECK_DISABLE 0x3	/* an enume */

/* The following are bit values to turn on or off options connected to the
 * malloc checking functionality */

/* Adds time to the memory checking information */
#define V_CRYPTO_MDEBUG_TIME	0x1 /* a bit */
/* Adds thread number to the memory checking information */
#define V_CRYPTO_MDEBUG_THREAD	0x2 /* a bit */

#define V_CRYPTO_MDEBUG_ALL (V_CRYPTO_MDEBUG_TIME | V_CRYPTO_MDEBUG_THREAD)


/* predec of the BIO type */
typedef struct bio_st BIO_dummy;

struct crypto_ex_data_st
	{
	STACK_OF(void) *sk;
	int dummy; /* gcc is screwing up this data structure :-( */
	};
DECLARE_STACK_OF(void)

/* This stuff is basically class callback functions
 * The current classes are SSL_CTX, SSL, SSL_SESSION, and a few more */

typedef struct crypto_ex_data_func_st
	{
	long argl;	/* Arbitary long */
	void *argp;	/* Arbitary void * */
	CRYPTO_EX_new *new_func;
	CRYPTO_EX_free *free_func;
	CRYPTO_EX_dup *dup_func;
	} CRYPTO_EX_DATA_FUNCS;

DECLARE_STACK_OF(CRYPTO_EX_DATA_FUNCS)

/* Per class, we have a STACK of CRYPTO_EX_DATA_FUNCS for each CRYPTO_EX_DATA
 * entry.
 */

#define CRYPTO_EX_INDEX_BIO		0
#define CRYPTO_EX_INDEX_SSL		1
#define CRYPTO_EX_INDEX_SSL_CTX		2
#define CRYPTO_EX_INDEX_SSL_SESSION	3
#define CRYPTO_EX_INDEX_X509_STORE	4
#define CRYPTO_EX_INDEX_X509_STORE_CTX	5
#define CRYPTO_EX_INDEX_RSA		6
#define CRYPTO_EX_INDEX_DSA		7
#define CRYPTO_EX_INDEX_DH		8
#define CRYPTO_EX_INDEX_ENGINE		9
#define CRYPTO_EX_INDEX_X509		10
#define CRYPTO_EX_INDEX_UI		11
#define CRYPTO_EX_INDEX_ECDSA		12
#define CRYPTO_EX_INDEX_ECDH		13
#define CRYPTO_EX_INDEX_COMP		14
#define CRYPTO_EX_INDEX_STORE		15

/* Dynamically assigned indexes start from this value (don't use directly, use
 * via CRYPTO_ex_data_new_class). */
#define CRYPTO_EX_INDEX_USER		100


/* This is the default callbacks, but we can have others as well:
 * this is needed in Win32 where the application malloc and the
 * library malloc may not be the same.
 */
#define CRYPTO_malloc_init()	CRYPTO_set_mem_functions(\
	malloc, realloc, free)

#if defined CRYPTO_MDEBUG_ALL || defined CRYPTO_MDEBUG_TIME || defined CRYPTO_MDEBUG_THREAD
# ifndef CRYPTO_MDEBUG /* avoid duplicate #define */
#  define CRYPTO_MDEBUG
# endif
#endif

/* Set standard debugging functions (not done by default
 * unless CRYPTO_MDEBUG is defined) */
#define CRYPTO_malloc_debug_init()	do {\
	CRYPTO_set_mem_debug_functions(\
		CRYPTO_dbg_malloc,\
		CRYPTO_dbg_realloc,\
		CRYPTO_dbg_free,\
		CRYPTO_dbg_set_options,\
		CRYPTO_dbg_get_options);\
	} while(0)

int CRYPTO_mem_ctrl(int mode);
int CRYPTO_is_mem_check_on(void);

/* for applications */
#define MemCheck_start() CRYPTO_mem_ctrl(CRYPTO_MEM_CHECK_ON)
#define MemCheck_stop()	CRYPTO_mem_ctrl(CRYPTO_MEM_CHECK_OFF)

/* for library-internal use */
#define MemCheck_on()	CRYPTO_mem_ctrl(CRYPTO_MEM_CHECK_ENABLE)
#define MemCheck_off()	CRYPTO_mem_ctrl(CRYPTO_MEM_CHECK_DISABLE)
#define is_MemCheck_on() CRYPTO_is_mem_check_on()

#define OPENSSL_malloc(num)	CRYPTO_malloc((int)num,__FILE__,__LINE__)
#define OPENSSL_strdup(str)	CRYPTO_strdup((str),__FILE__,__LINE__)
#define OPENSSL_realloc(addr,num) \
	CRYPTO_realloc((char *)addr,(int)num,__FILE__,__LINE__)
#define OPENSSL_realloc_clean(addr,old_num,num) \
	CRYPTO_realloc_clean(addr,old_num,num,__FILE__,__LINE__)
#define OPENSSL_remalloc(addr,num) \
	CRYPTO_remalloc((char **)addr,(int)num,__FILE__,__LINE__)
#define OPENSSL_freeFunc	CRYPTO_free
#define OPENSSL_free(addr)	CRYPTO_free(addr)

#define OPENSSL_malloc_locked(num) \
	CRYPTO_malloc_locked((int)num,__FILE__,__LINE__)
#define OPENSSL_free_locked(addr) CRYPTO_free_locked(addr)


const char *SSLeay_version(int type);
unsigned long SSLeay(void);

int OPENSSL_issetugid(void);

/* An opaque type representing an implementation of "ex_data" support */
typedef struct st_CRYPTO_EX_DATA_IMPL	CRYPTO_EX_DATA_IMPL;
/* Return an opaque pointer to the current "ex_data" implementation */
const CRYPTO_EX_DATA_IMPL *CRYPTO_get_ex_data_implementation(void);
/* Sets the "ex_data" implementation to be used (if it's not too late) */
int CRYPTO_set_ex_data_implementation(const CRYPTO_EX_DATA_IMPL *i);
/* Get a new "ex_data" class, and return the corresponding "class_index" */
int CRYPTO_ex_data_new_class(void);
/* Within a given class, get/register a new index */
int CRYPTO_get_ex_new_index(int class_index, long argl, void *argp,
		CRYPTO_EX_new *new_func, CRYPTO_EX_dup *dup_func,
		CRYPTO_EX_free *free_func);
/* Initialise/duplicate/free CRYPTO_EX_DATA variables corresponding to a given
 * class (invokes whatever per-class callbacks are applicable) */
int CRYPTO_new_ex_data(int class_index, void *obj, CRYPTO_EX_DATA *ad);
int CRYPTO_dup_ex_data(int class_index, CRYPTO_EX_DATA *to,
		CRYPTO_EX_DATA *from);
void CRYPTO_free_ex_data(int class_index, void *obj, CRYPTO_EX_DATA *ad);
/* Get/set data in a CRYPTO_EX_DATA variable corresponding to a particular index
 * (relative to the class type involved) */
int CRYPTO_set_ex_data(CRYPTO_EX_DATA *ad, int idx, void *val);
void *CRYPTO_get_ex_data(const CRYPTO_EX_DATA *ad,int idx);
/* This function cleans up all "ex_data" state. It mustn't be called under
 * potential race-conditions. */
void CRYPTO_cleanup_all_ex_data(void);

int CRYPTO_get_new_lockid(char *name);

int CRYPTO_num_locks(void); /* return CRYPTO_NUM_LOCKS (shared libs!) */
void CRYPTO_lock(int mode, int type,const char *file,int line);
void CRYPTO_set_locking_callback(void (*func)(int mode,int type,
					      const char *file,int line));
void (*CRYPTO_get_locking_callback(void))(int mode,int type,const char *file,
		int line);
void CRYPTO_set_add_lock_callback(int (*func)(int *num,int mount,int type,
					      const char *file, int line));
int (*CRYPTO_get_add_lock_callback(void))(int *num,int mount,int type,
					  const char *file,int line);

/* Don't use this structure directly. */
typedef struct crypto_threadid_st
	{
	void *ptr;
	unsigned long val;
	} CRYPTO_THREADID;
/* Only use CRYPTO_THREADID_set_[numeric|pointer]() within callbacks */
void CRYPTO_THREADID_set_numeric(CRYPTO_THREADID *id, unsigned long val);
void CRYPTO_THREADID_set_pointer(CRYPTO_THREADID *id, void *ptr);
int CRYPTO_THREADID_set_callback(void (*threadid_func)(CRYPTO_THREADID *));
void (*CRYPTO_THREADID_get_callback(void))(CRYPTO_THREADID *);
void CRYPTO_THREADID_current(CRYPTO_THREADID *id);
int CRYPTO_THREADID_cmp(const CRYPTO_THREADID *a, const CRYPTO_THREADID *b);
void CRYPTO_THREADID_cpy(CRYPTO_THREADID *dest, const CRYPTO_THREADID *src);
unsigned long CRYPTO_THREADID_hash(const CRYPTO_THREADID *id);
#ifndef OPENSSL_NO_DEPRECATED
void CRYPTO_set_id_callback(unsigned long (*func)(void));
unsigned long (*CRYPTO_get_id_callback(void))(void);
unsigned long CRYPTO_thread_id(void);
#endif

const char *CRYPTO_get_lock_name(int type);
int CRYPTO_add_lock(int *pointer,int amount,int type, const char *file,
		    int line);

int CRYPTO_get_new_dynlockid(void);
void CRYPTO_destroy_dynlockid(int i);
struct CRYPTO_dynlock_value *CRYPTO_get_dynlock_value(int i);
void CRYPTO_set_dynlock_create_callback(struct CRYPTO_dynlock_value *(*dyn_create_function)(const char *file, int line));
void CRYPTO_set_dynlock_lock_callback(void (*dyn_lock_function)(int mode, struct CRYPTO_dynlock_value *l, const char *file, int line));
void CRYPTO_set_dynlock_destroy_callback(void (*dyn_destroy_function)(struct CRYPTO_dynlock_value *l, const char *file, int line));
struct CRYPTO_dynlock_value *(*CRYPTO_get_dynlock_create_callback(void))(const char *file,int line);
void (*CRYPTO_get_dynlock_lock_callback(void))(int mode, struct CRYPTO_dynlock_value *l, const char *file,int line);
void (*CRYPTO_get_dynlock_destroy_callback(void))(struct CRYPTO_dynlock_value *l, const char *file,int line);

/* CRYPTO_set_mem_functions includes CRYPTO_set_locked_mem_functions --
 * call the latter last if you need different functions */
int CRYPTO_set_mem_functions(void *(*m)(size_t),void *(*r)(void *,size_t), void (*f)(void *));
int CRYPTO_set_locked_mem_functions(void *(*m)(size_t), void (*free_func)(void *));
int CRYPTO_set_mem_ex_functions(void *(*m)(size_t,const char *,int),
                                void *(*r)(void *,size_t,const char *,int),
                                void (*f)(void *));
int CRYPTO_set_locked_mem_ex_functions(void *(*m)(size_t,const char *,int),
                                       void (*free_func)(void *));
int CRYPTO_set_mem_debug_functions(void (*m)(void *,int,const char *,int,int),
				   void (*r)(void *,void *,int,const char *,int,int),
				   void (*f)(void *,int),
				   void (*so)(long),
				   long (*go)(void));
void CRYPTO_get_mem_functions(void *(**m)(size_t),void *(**r)(void *, size_t), void (**f)(void *));
void CRYPTO_get_locked_mem_functions(void *(**m)(size_t), void (**f)(void *));
void CRYPTO_get_mem_ex_functions(void *(**m)(size_t,const char *,int),
                                 void *(**r)(void *, size_t,const char *,int),
                                 void (**f)(void *));
void CRYPTO_get_locked_mem_ex_functions(void *(**m)(size_t,const char *,int),
                                        void (**f)(void *));
void CRYPTO_get_mem_debug_functions(void (**m)(void *,int,const char *,int,int),
				    void (**r)(void *,void *,int,const char *,int,int),
				    void (**f)(void *,int),
				    void (**so)(long),
				    long (**go)(void));

void *CRYPTO_malloc_locked(int num, const char *file, int line);
void CRYPTO_free_locked(void *ptr);
void *CRYPTO_malloc(int num, const char *file, int line);
char *CRYPTO_strdup(const char *str, const char *file, int line);
void CRYPTO_free(void *ptr);
void *CRYPTO_realloc(void *addr,int num, const char *file, int line);
void *CRYPTO_realloc_clean(void *addr,int old_num,int num,const char *file,
			   int line);
void *CRYPTO_remalloc(void *addr,int num, const char *file, int line);

void OPENSSL_cleanse(void *ptr, size_t len);

void CRYPTO_set_mem_debug_options(long bits);
long CRYPTO_get_mem_debug_options(void);

#define CRYPTO_push_info(info) \
        CRYPTO_push_info_(info, __FILE__, __LINE__);
int CRYPTO_push_info_(const char *info, const char *file, int line);
int CRYPTO_pop_info(void);
int CRYPTO_remove_all_info(void);


/* Default debugging functions (enabled by CRYPTO_malloc_debug_init() macro;
 * used as default in CRYPTO_MDEBUG compilations): */
/* The last argument has the following significance:
 *
 * 0:	called before the actual memory allocation has taken place
 * 1:	called after the actual memory allocation has taken place
 */
void CRYPTO_dbg_malloc(void *addr,int num,const char *file,int line,int before_p);
void CRYPTO_dbg_realloc(void *addr1,void *addr2,int num,const char *file,int line,int before_p);
void CRYPTO_dbg_free(void *addr,int before_p);
/* Tell the debugging code about options.  By default, the following values
 * apply:
 *
 * 0:                           Clear all options.
 * V_CRYPTO_MDEBUG_TIME (1):    Set the "Show Time" option.
 * V_CRYPTO_MDEBUG_THREAD (2):  Set the "Show Thread Number" option.
 * V_CRYPTO_MDEBUG_ALL (3):     1 + 2
 */
void CRYPTO_dbg_set_options(long bits);
long CRYPTO_dbg_get_options(void);


#ifndef OPENSSL_NO_FP_API
void CRYPTO_mem_leaks_fp(FILE *);
#endif
void CRYPTO_mem_leaks(struct bio_st *bio);
/* unsigned long order, char *file, int line, int num_bytes, char *addr */
typedef void *CRYPTO_MEM_LEAK_CB(unsigned long, const char *, int, int, void *);
void CRYPTO_mem_leaks_cb(CRYPTO_MEM_LEAK_CB *cb);

/* die if we have to */
void OpenSSLDie(const char *file,int line,const char *assertion);
#define OPENSSL_assert(e)       (void)((e) ? 0 : (OpenSSLDie(__FILE__, __LINE__, #e),1))

unsigned long *OPENSSL_ia32cap_loc(void);
#define OPENSSL_ia32cap (*(OPENSSL_ia32cap_loc()))
=======
# if OPENSSL_API_COMPAT < 0x10100000L
#  define SSLeay                  OpenSSL_version_num
#  define SSLeay_version          OpenSSL_version
#  define SSLEAY_VERSION_NUMBER   OPENSSL_VERSION_NUMBER
#  define SSLEAY_VERSION          OPENSSL_VERSION
#  define SSLEAY_CFLAGS           OPENSSL_CFLAGS
#  define SSLEAY_BUILT_ON         OPENSSL_BUILT_ON
#  define SSLEAY_PLATFORM         OPENSSL_PLATFORM
#  define SSLEAY_DIR              OPENSSL_DIR

/*
 * Old type for allocating dynamic locks. No longer used. Use the new thread
 * API instead.
 */
typedef struct {
    int dummy;
} CRYPTO_dynlock;

# endif /* OPENSSL_API_COMPAT */

typedef void CRYPTO_RWLOCK;

CRYPTO_RWLOCK *CRYPTO_THREAD_lock_new(void);
int CRYPTO_THREAD_read_lock(CRYPTO_RWLOCK *lock);
int CRYPTO_THREAD_write_lock(CRYPTO_RWLOCK *lock);
int CRYPTO_THREAD_unlock(CRYPTO_RWLOCK *lock);
void CRYPTO_THREAD_lock_free(CRYPTO_RWLOCK *lock);

int CRYPTO_atomic_add(int *val, int amount, int *ret, CRYPTO_RWLOCK *lock);

/*
 * The following can be used to detect memory leaks in the library. If
 * used, it turns on malloc checking
 */
# define CRYPTO_MEM_CHECK_OFF     0x0   /* Control only */
# define CRYPTO_MEM_CHECK_ON      0x1   /* Control and mode bit */
# define CRYPTO_MEM_CHECK_ENABLE  0x2   /* Control and mode bit */
# define CRYPTO_MEM_CHECK_DISABLE 0x3   /* Control only */

struct crypto_ex_data_st {
    STACK_OF(void) *sk;
};
DEFINE_STACK_OF(void)

/*
 * Per class, we have a STACK of function pointers.
 */
# define CRYPTO_EX_INDEX_SSL              0
# define CRYPTO_EX_INDEX_SSL_CTX          1
# define CRYPTO_EX_INDEX_SSL_SESSION      2
# define CRYPTO_EX_INDEX_X509             3
# define CRYPTO_EX_INDEX_X509_STORE       4
# define CRYPTO_EX_INDEX_X509_STORE_CTX   5
# define CRYPTO_EX_INDEX_DH               6
# define CRYPTO_EX_INDEX_DSA              7
# define CRYPTO_EX_INDEX_EC_KEY           8
# define CRYPTO_EX_INDEX_RSA              9
# define CRYPTO_EX_INDEX_ENGINE          10
# define CRYPTO_EX_INDEX_UI              11
# define CRYPTO_EX_INDEX_BIO             12
# define CRYPTO_EX_INDEX_APP             13
# define CRYPTO_EX_INDEX_UI_METHOD       14
# define CRYPTO_EX_INDEX_DRBG            15
# define CRYPTO_EX_INDEX__COUNT          16

/* No longer needed, so this is a no-op */
#define OPENSSL_malloc_init() while(0) continue

int CRYPTO_mem_ctrl(int mode);

# define OPENSSL_malloc(num) \
        CRYPTO_malloc(num, OPENSSL_FILE, OPENSSL_LINE)
# define OPENSSL_zalloc(num) \
        CRYPTO_zalloc(num, OPENSSL_FILE, OPENSSL_LINE)
# define OPENSSL_realloc(addr, num) \
        CRYPTO_realloc(addr, num, OPENSSL_FILE, OPENSSL_LINE)
# define OPENSSL_clear_realloc(addr, old_num, num) \
        CRYPTO_clear_realloc(addr, old_num, num, OPENSSL_FILE, OPENSSL_LINE)
# define OPENSSL_clear_free(addr, num) \
        CRYPTO_clear_free(addr, num, OPENSSL_FILE, OPENSSL_LINE)
# define OPENSSL_free(addr) \
        CRYPTO_free(addr, OPENSSL_FILE, OPENSSL_LINE)
# define OPENSSL_memdup(str, s) \
        CRYPTO_memdup((str), s, OPENSSL_FILE, OPENSSL_LINE)
# define OPENSSL_strdup(str) \
        CRYPTO_strdup(str, OPENSSL_FILE, OPENSSL_LINE)
# define OPENSSL_strndup(str, n) \
        CRYPTO_strndup(str, n, OPENSSL_FILE, OPENSSL_LINE)
# define OPENSSL_secure_malloc(num) \
        CRYPTO_secure_malloc(num, OPENSSL_FILE, OPENSSL_LINE)
# define OPENSSL_secure_zalloc(num) \
        CRYPTO_secure_zalloc(num, OPENSSL_FILE, OPENSSL_LINE)
# define OPENSSL_secure_free(addr) \
        CRYPTO_secure_free(addr, OPENSSL_FILE, OPENSSL_LINE)
# define OPENSSL_secure_clear_free(addr, num) \
        CRYPTO_secure_clear_free(addr, num, OPENSSL_FILE, OPENSSL_LINE)
# define OPENSSL_secure_actual_size(ptr) \
        CRYPTO_secure_actual_size(ptr)

size_t OPENSSL_strlcpy(char *dst, const char *src, size_t siz);
size_t OPENSSL_strlcat(char *dst, const char *src, size_t siz);
size_t OPENSSL_strnlen(const char *str, size_t maxlen);
char *OPENSSL_buf2hexstr(const unsigned char *buffer, long len);
unsigned char *OPENSSL_hexstr2buf(const char *str, long *len);
int OPENSSL_hexchar2int(unsigned char c);

# define OPENSSL_MALLOC_MAX_NELEMS(type)  (((1U<<(sizeof(int)*8-1))-1)/sizeof(type))

unsigned long OpenSSL_version_num(void);
const char *OpenSSL_version(int type);
# define OPENSSL_VERSION          0
# define OPENSSL_CFLAGS           1
# define OPENSSL_BUILT_ON         2
# define OPENSSL_PLATFORM         3
# define OPENSSL_DIR              4
# define OPENSSL_ENGINES_DIR      5

int OPENSSL_issetugid(void);

typedef void CRYPTO_EX_new (void *parent, void *ptr, CRYPTO_EX_DATA *ad,
                           int idx, long argl, void *argp);
typedef void CRYPTO_EX_free (void *parent, void *ptr, CRYPTO_EX_DATA *ad,
                             int idx, long argl, void *argp);
typedef int CRYPTO_EX_dup (CRYPTO_EX_DATA *to, const CRYPTO_EX_DATA *from,
                           void *from_d, int idx, long argl, void *argp);
__owur int CRYPTO_get_ex_new_index(int class_index, long argl, void *argp,
                            CRYPTO_EX_new *new_func, CRYPTO_EX_dup *dup_func,
                            CRYPTO_EX_free *free_func);
/* No longer use an index. */
int CRYPTO_free_ex_index(int class_index, int idx);

/*
 * Initialise/duplicate/free CRYPTO_EX_DATA variables corresponding to a
 * given class (invokes whatever per-class callbacks are applicable)
 */
int CRYPTO_new_ex_data(int class_index, void *obj, CRYPTO_EX_DATA *ad);
int CRYPTO_dup_ex_data(int class_index, CRYPTO_EX_DATA *to,
                       const CRYPTO_EX_DATA *from);

void CRYPTO_free_ex_data(int class_index, void *obj, CRYPTO_EX_DATA *ad);

/*
 * Get/set data in a CRYPTO_EX_DATA variable corresponding to a particular
 * index (relative to the class type involved)
 */
int CRYPTO_set_ex_data(CRYPTO_EX_DATA *ad, int idx, void *val);
void *CRYPTO_get_ex_data(const CRYPTO_EX_DATA *ad, int idx);

# if OPENSSL_API_COMPAT < 0x10100000L
/*
 * This function cleans up all "ex_data" state. It mustn't be called under
 * potential race-conditions.
 */
# define CRYPTO_cleanup_all_ex_data() while(0) continue

/*
 * The old locking functions have been removed completely without compatibility
 * macros. This is because the old functions either could not properly report
 * errors, or the returned error values were not clearly documented.
 * Replacing the locking functions with no-ops would cause race condition
 * issues in the affected applications. It is far better for them to fail at
 * compile time.
 * On the other hand, the locking callbacks are no longer used.  Consequently,
 * the callback management functions can be safely replaced with no-op macros.
 */
#  define CRYPTO_num_locks()            (1)
#  define CRYPTO_set_locking_callback(func)
#  define CRYPTO_get_locking_callback()         (NULL)
#  define CRYPTO_set_add_lock_callback(func)
#  define CRYPTO_get_add_lock_callback()        (NULL)

/*
 * These defines where used in combination with the old locking callbacks,
 * they are not called anymore, but old code that's not called might still
 * use them.
 */
#  define CRYPTO_LOCK             1
#  define CRYPTO_UNLOCK           2
#  define CRYPTO_READ             4
#  define CRYPTO_WRITE            8

/* This structure is no longer used */
typedef struct crypto_threadid_st {
    int dummy;
} CRYPTO_THREADID;
/* Only use CRYPTO_THREADID_set_[numeric|pointer]() within callbacks */
#  define CRYPTO_THREADID_set_numeric(id, val)
#  define CRYPTO_THREADID_set_pointer(id, ptr)
#  define CRYPTO_THREADID_set_callback(threadid_func)   (0)
#  define CRYPTO_THREADID_get_callback()                (NULL)
#  define CRYPTO_THREADID_current(id)
#  define CRYPTO_THREADID_cmp(a, b)                     (-1)
#  define CRYPTO_THREADID_cpy(dest, src)
#  define CRYPTO_THREADID_hash(id)                      (0UL)

#  if OPENSSL_API_COMPAT < 0x10000000L
#   define CRYPTO_set_id_callback(func)
#   define CRYPTO_get_id_callback()                     (NULL)
#   define CRYPTO_thread_id()                           (0UL)
#  endif /* OPENSSL_API_COMPAT < 0x10000000L */

#  define CRYPTO_set_dynlock_create_callback(dyn_create_function)
#  define CRYPTO_set_dynlock_lock_callback(dyn_lock_function)
#  define CRYPTO_set_dynlock_destroy_callback(dyn_destroy_function)
#  define CRYPTO_get_dynlock_create_callback()          (NULL)
#  define CRYPTO_get_dynlock_lock_callback()            (NULL)
#  define CRYPTO_get_dynlock_destroy_callback()         (NULL)
# endif /* OPENSSL_API_COMPAT < 0x10100000L */

int CRYPTO_set_mem_functions(
        void *(*m) (size_t, const char *, int),
        void *(*r) (void *, size_t, const char *, int),
        void (*f) (void *, const char *, int));
int CRYPTO_set_mem_debug(int flag);
void CRYPTO_get_mem_functions(
        void *(**m) (size_t, const char *, int),
        void *(**r) (void *, size_t, const char *, int),
        void (**f) (void *, const char *, int));

void *CRYPTO_malloc(size_t num, const char *file, int line);
void *CRYPTO_zalloc(size_t num, const char *file, int line);
void *CRYPTO_memdup(const void *str, size_t siz, const char *file, int line);
char *CRYPTO_strdup(const char *str, const char *file, int line);
char *CRYPTO_strndup(const char *str, size_t s, const char *file, int line);
void CRYPTO_free(void *ptr, const char *file, int line);
void CRYPTO_clear_free(void *ptr, size_t num, const char *file, int line);
void *CRYPTO_realloc(void *addr, size_t num, const char *file, int line);
void *CRYPTO_clear_realloc(void *addr, size_t old_num, size_t num,
                           const char *file, int line);

int CRYPTO_secure_malloc_init(size_t sz, int minsize);
int CRYPTO_secure_malloc_done(void);
void *CRYPTO_secure_malloc(size_t num, const char *file, int line);
void *CRYPTO_secure_zalloc(size_t num, const char *file, int line);
void CRYPTO_secure_free(void *ptr, const char *file, int line);
void CRYPTO_secure_clear_free(void *ptr, size_t num,
                              const char *file, int line);
int CRYPTO_secure_allocated(const void *ptr);
int CRYPTO_secure_malloc_initialized(void);
size_t CRYPTO_secure_actual_size(void *ptr);
size_t CRYPTO_secure_used(void);

void OPENSSL_cleanse(void *ptr, size_t len);

# ifndef OPENSSL_NO_CRYPTO_MDEBUG
#  define OPENSSL_mem_debug_push(info) \
        CRYPTO_mem_debug_push(info, OPENSSL_FILE, OPENSSL_LINE)
#  define OPENSSL_mem_debug_pop() \
        CRYPTO_mem_debug_pop()
int CRYPTO_mem_debug_push(const char *info, const char *file, int line);
int CRYPTO_mem_debug_pop(void);
void CRYPTO_get_alloc_counts(int *mcount, int *rcount, int *fcount);

/*-
 * Debugging functions (enabled by CRYPTO_set_mem_debug(1))
 * The flag argument has the following significance:
 *   0:   called before the actual memory allocation has taken place
 *   1:   called after the actual memory allocation has taken place
 */
void CRYPTO_mem_debug_malloc(void *addr, size_t num, int flag,
        const char *file, int line);
void CRYPTO_mem_debug_realloc(void *addr1, void *addr2, size_t num, int flag,
        const char *file, int line);
void CRYPTO_mem_debug_free(void *addr, int flag,
        const char *file, int line);

int CRYPTO_mem_leaks_cb(int (*cb) (const char *str, size_t len, void *u),
                        void *u);
#  ifndef OPENSSL_NO_STDIO
int CRYPTO_mem_leaks_fp(FILE *);
#  endif
int CRYPTO_mem_leaks(BIO *bio);
# endif

/* die if we have to */
ossl_noreturn void OPENSSL_die(const char *assertion, const char *file, int line);
# if OPENSSL_API_COMPAT < 0x10100000L
#  define OpenSSLDie(f,l,a) OPENSSL_die((a),(f),(l))
# endif
# define OPENSSL_assert(e) \
    (void)((e) ? 0 : (OPENSSL_die("assertion failed: " #e, OPENSSL_FILE, OPENSSL_LINE), 1))

>>>>>>> Stashed changes
int OPENSSL_isservice(void);

int FIPS_mode(void);
int FIPS_mode_set(int r);

void OPENSSL_init(void);
<<<<<<< Updated upstream

#define fips_md_init(alg) fips_md_init_ctx(alg, alg)

#ifdef OPENSSL_FIPS
#define fips_md_init_ctx(alg, cx) \
	int alg##_Init(cx##_CTX *c) \
	{ \
	if (FIPS_mode()) OpenSSLDie(__FILE__, __LINE__, \
		"Low level API call to digest " #alg " forbidden in FIPS mode!"); \
	return private_##alg##_Init(c); \
	} \
	int private_##alg##_Init(cx##_CTX *c)

#define fips_cipher_abort(alg) \
	if (FIPS_mode()) OpenSSLDie(__FILE__, __LINE__, \
		"Low level API call to cipher " #alg " forbidden in FIPS mode!")

#else
#define fips_md_init_ctx(alg, cx) \
	int alg##_Init(cx##_CTX *c)
#define fips_cipher_abort(alg) while(0)
#endif

/* CRYPTO_memcmp returns zero iff the |len| bytes at |a| and |b| are equal. It
 * takes an amount of time dependent on |len|, but independent of the contents
 * of |a| and |b|. Unlike memcmp, it cannot be used to put elements into a
 * defined order as the return value when a != b is undefined, other than to be
 * non-zero. */
int CRYPTO_memcmp(const void *a, const void *b, size_t len);

/* BEGIN ERROR CODES */
/* The following lines are auto generated by the script mkerr.pl. Any changes
 * made after this point may be overwritten when the script is next run.
 */
void ERR_load_CRYPTO_strings(void);

/* Error codes for the CRYPTO functions. */

/* Function codes. */
#define CRYPTO_F_CRYPTO_GET_EX_NEW_INDEX		 100
#define CRYPTO_F_CRYPTO_GET_NEW_DYNLOCKID		 103
#define CRYPTO_F_CRYPTO_GET_NEW_LOCKID			 101
#define CRYPTO_F_CRYPTO_SET_EX_DATA			 102
#define CRYPTO_F_DEF_ADD_INDEX				 104
#define CRYPTO_F_DEF_GET_CLASS				 105
#define CRYPTO_F_FIPS_MODE_SET				 109
#define CRYPTO_F_INT_DUP_EX_DATA			 106
#define CRYPTO_F_INT_FREE_EX_DATA			 107
#define CRYPTO_F_INT_NEW_EX_DATA			 108

/* Reason codes. */
#define CRYPTO_R_FIPS_MODE_NOT_SUPPORTED		 101
#define CRYPTO_R_NO_DYNLOCK_CREATE_CALLBACK		 100

#ifdef  __cplusplus
}
#endif
=======
# ifdef OPENSSL_SYS_UNIX
void OPENSSL_fork_prepare(void);
void OPENSSL_fork_parent(void);
void OPENSSL_fork_child(void);
# endif

struct tm *OPENSSL_gmtime(const time_t *timer, struct tm *result);
int OPENSSL_gmtime_adj(struct tm *tm, int offset_day, long offset_sec);
int OPENSSL_gmtime_diff(int *pday, int *psec,
                        const struct tm *from, const struct tm *to);

/*
 * CRYPTO_memcmp returns zero iff the |len| bytes at |a| and |b| are equal.
 * It takes an amount of time dependent on |len|, but independent of the
 * contents of |a| and |b|. Unlike memcmp, it cannot be used to put elements
 * into a defined order as the return value when a != b is undefined, other
 * than to be non-zero.
 */
int CRYPTO_memcmp(const void * in_a, const void * in_b, size_t len);

/* Standard initialisation options */
# define OPENSSL_INIT_NO_LOAD_CRYPTO_STRINGS 0x00000001L
# define OPENSSL_INIT_LOAD_CRYPTO_STRINGS    0x00000002L
# define OPENSSL_INIT_ADD_ALL_CIPHERS        0x00000004L
# define OPENSSL_INIT_ADD_ALL_DIGESTS        0x00000008L
# define OPENSSL_INIT_NO_ADD_ALL_CIPHERS     0x00000010L
# define OPENSSL_INIT_NO_ADD_ALL_DIGESTS     0x00000020L
# define OPENSSL_INIT_LOAD_CONFIG            0x00000040L
# define OPENSSL_INIT_NO_LOAD_CONFIG         0x00000080L
# define OPENSSL_INIT_ASYNC                  0x00000100L
# define OPENSSL_INIT_ENGINE_RDRAND          0x00000200L
# define OPENSSL_INIT_ENGINE_DYNAMIC         0x00000400L
# define OPENSSL_INIT_ENGINE_OPENSSL         0x00000800L
# define OPENSSL_INIT_ENGINE_CRYPTODEV       0x00001000L
# define OPENSSL_INIT_ENGINE_CAPI            0x00002000L
# define OPENSSL_INIT_ENGINE_PADLOCK         0x00004000L
# define OPENSSL_INIT_ENGINE_AFALG           0x00008000L
/* OPENSSL_INIT_ZLIB                         0x00010000L */
# define OPENSSL_INIT_ATFORK                 0x00020000L
/* OPENSSL_INIT_BASE_ONLY                    0x00040000L */
# define OPENSSL_INIT_NO_ATEXIT              0x00080000L
/* OPENSSL_INIT flag range 0xfff00000 reserved for OPENSSL_init_ssl() */
/* Max OPENSSL_INIT flag value is 0x80000000 */

/* openssl and dasync not counted as builtin */
# define OPENSSL_INIT_ENGINE_ALL_BUILTIN \
    (OPENSSL_INIT_ENGINE_RDRAND | OPENSSL_INIT_ENGINE_DYNAMIC \
    | OPENSSL_INIT_ENGINE_CRYPTODEV | OPENSSL_INIT_ENGINE_CAPI | \
    OPENSSL_INIT_ENGINE_PADLOCK)


/* Library initialisation functions */
void OPENSSL_cleanup(void);
int OPENSSL_init_crypto(uint64_t opts, const OPENSSL_INIT_SETTINGS *settings);
int OPENSSL_atexit(void (*handler)(void));
void OPENSSL_thread_stop(void);

/* Low-level control of initialization */
OPENSSL_INIT_SETTINGS *OPENSSL_INIT_new(void);
# ifndef OPENSSL_NO_STDIO
int OPENSSL_INIT_set_config_filename(OPENSSL_INIT_SETTINGS *settings,
                                     const char *config_filename);
void OPENSSL_INIT_set_config_file_flags(OPENSSL_INIT_SETTINGS *settings,
                                        unsigned long flags);
int OPENSSL_INIT_set_config_appname(OPENSSL_INIT_SETTINGS *settings,
                                    const char *config_appname);
# endif
void OPENSSL_INIT_free(OPENSSL_INIT_SETTINGS *settings);

# if defined(OPENSSL_THREADS) && !defined(CRYPTO_TDEBUG)
#  if defined(_WIN32)
#   if defined(BASETYPES) || defined(_WINDEF_H)
/* application has to include <windows.h> in order to use this */
typedef DWORD CRYPTO_THREAD_LOCAL;
typedef DWORD CRYPTO_THREAD_ID;

typedef LONG CRYPTO_ONCE;
#    define CRYPTO_ONCE_STATIC_INIT 0
#   endif
#  else
#   include <pthread.h>
typedef pthread_once_t CRYPTO_ONCE;
typedef pthread_key_t CRYPTO_THREAD_LOCAL;
typedef pthread_t CRYPTO_THREAD_ID;

#   define CRYPTO_ONCE_STATIC_INIT PTHREAD_ONCE_INIT
#  endif
# endif

# if !defined(CRYPTO_ONCE_STATIC_INIT)
typedef unsigned int CRYPTO_ONCE;
typedef unsigned int CRYPTO_THREAD_LOCAL;
typedef unsigned int CRYPTO_THREAD_ID;
#  define CRYPTO_ONCE_STATIC_INIT 0
# endif

int CRYPTO_THREAD_run_once(CRYPTO_ONCE *once, void (*init)(void));

int CRYPTO_THREAD_init_local(CRYPTO_THREAD_LOCAL *key, void (*cleanup)(void *));
void *CRYPTO_THREAD_get_local(CRYPTO_THREAD_LOCAL *key);
int CRYPTO_THREAD_set_local(CRYPTO_THREAD_LOCAL *key, void *val);
int CRYPTO_THREAD_cleanup_local(CRYPTO_THREAD_LOCAL *key);

CRYPTO_THREAD_ID CRYPTO_THREAD_get_current_id(void);
int CRYPTO_THREAD_compare_id(CRYPTO_THREAD_ID a, CRYPTO_THREAD_ID b);


# ifdef  __cplusplus
}
# endif
>>>>>>> Stashed changes
#endif
