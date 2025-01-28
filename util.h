#pragma once

#include <stdio.h>

#define MSG(fmt, ...) fprintf(stderr, "%s(%d) in %s(): " fmt "\n", __FILE__, __LINE__, __func__, ##__VA_ARGS__)

// ERR() macro reports the error. In C we just print it out. In Python we set
// the Python error
#ifndef ERR
  #define ERR(fmt, ...) MSG("Failure!!! " fmt, ##__VA_ARGS__)
#endif

#define try(expr, ...) do {                                     \
        if(ctx->verbose)                                        \
            MSG("ctx=%p: Evaluating   '" #expr "'", ctx);       \
        if(!(expr))                                             \
        {                                                       \
            ERR("'" #expr "' is false" __VA_ARGS__);            \
            goto done;                                          \
        }                                                       \
    } while(0)

#define try_arv(expr) do {                              \
        if(ctx->verbose)                                \
            MSG("ctx=%p: Calling   '" #expr "'", ctx);  \
        expr;                                           \
        if(error != NULL)                               \
        {                                               \
            ERR("'" #expr "' produced '%s'",            \
                error->message);                        \
            g_clear_error(&error);                      \
            goto done;                                  \
        }                                               \
    } while(0)

#define try_arv_extra_reporting(expr, extra_verbose_before, extra_verbose_after, extra_err) do { \
        if(ctx->verbose)                                                \
        {                                                               \
            extra_verbose_before;                                       \
            MSG("ctx=%p: Calling   '" #expr "'", ctx);                  \
        }                                                               \
        expr;                                                           \
        if(ctx->verbose)                                                \
        {                                                               \
            extra_verbose_after;                                        \
        }                                                               \
        if(error != NULL)                                               \
        {                                                               \
            ERR("'" #expr "' produced '%s'",                            \
                error->message);                                        \
            extra_err;                                                  \
            g_clear_error(&error);                                      \
            goto done;                                                  \
        }                                                               \
    } while(0)

// THIS MACRO MAY LEAVE error ALLOCATED. YOU NEED TO g_clear_error() yourself
#define try_arv_or(expr, condition) do {                                \
        if(ctx->verbose)                                                \
            MSG("ctx=%p: Calling   '" #expr "'", ctx);                  \
        expr;                                                           \
        if(error != NULL)                                               \
        {                                                               \
            if(!(condition))                                            \
            {                                                           \
                ERR("'" #expr "' produced '%s'",                        \
                    error->message);                                    \
                g_clear_error(&error);                                  \
                goto done;                                              \
            }                                                           \
            else if(ctx->verbose)                                       \
                MSG("  failed ('%s'), but extra condition '" #condition "' is true, so this failure is benign", \
                    error->message);                                    \
        }                                                               \
    } while(0)

#define try_arv_and(expr, condition) do {                               \
        if(ctx->verbose)                                                \
            MSG("ctx=%p: Calling   '" #expr "'", ctx);                  \
        expr;                                                           \
        if(error != NULL)                                               \
        {                                                               \
            ERR("'" #expr "' produced '%s'",                            \
                error->message);                                        \
            g_clear_error(&error);                                      \
            goto done;                                                  \
        }                                                               \
        if(!(condition))                                                \
        {                                                               \
            ERR("'" #expr "' produced no error, but the extra condition '" #condition "' failed"); \
            goto done;                                                  \
        }                                                               \
    } while(0)
